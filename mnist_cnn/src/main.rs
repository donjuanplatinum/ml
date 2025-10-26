use anyhow::Result;
use candle_nn::{conv2d, linear, loss,  AdamW, Conv2d, Conv2dConfig, Linear, Optimizer, VarBuilder, VarMap};
use candle_core::{DType, Device, Tensor, D};
use candle_datasets::vision;
use rand::{seq::SliceRandom, thread_rng};
use image::ImageReader;

use clap::{Parser,Subcommand};

#[derive(Parser)]
#[command(name = "ml")]
#[command(about = "Mnist CNN", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        #[arg(short, long, default_value_t = 10)]
        epochs: usize,
	#[arg(short, long, default_value_t = 0.01)]
	learning_rate: f64,
	#[arg(short,long,default_value_t = 32)]
	batch_size: usize,
    },
    Infer {
        #[arg(short, long)]
        image_path: String,
	#[arg(short,long)]
	model_path: String,
    },
}

pub struct MnistCnn{
    conv1:  Conv2d,
    conv2:  Conv2d,
    fc1: Linear,
    fc2: Linear,
}

impl MnistCnn{
    pub fn new(vb: VarBuilder) -> Result<Self>{
	let convdefault = Conv2dConfig{
	    padding: 1,
	    ..Default::default()
	};
	let conv1 = conv2d(1,32,3,convdefault,vb.pp("c1"))?;
	let conv2 = conv2d(32,64,3,convdefault,vb.pp("c2"))?;
	let fc1 = linear(64*7*7,128,vb.pp("fc1"))?;
	let fc2 = linear(128,10,vb.pp("fc2"))?;
	Ok(Self{
	    conv1,
	    conv2,
	    fc1,
	    fc2,
	})
    }
}
impl MnistCnn {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let batch_size = xs.dim(0)?; 
        let xs = xs.reshape((batch_size, 1, 28, 28))?;
        Ok(xs
            .apply(&self.conv1)?
            .relu()?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .relu()?
            .max_pool2d(2)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?
            .apply(&self.fc2)?)
    }
}

fn infer(image_path: &str,model_path: &str) -> Result<()> {
    
    let model_file = [
	model_path,
	    ];
    let vb = unsafe {
	VarBuilder::from_mmaped_safetensors(&model_file,DType::F32,&Device::Cpu)?
    };
    
    let model = MnistCnn::new(vb)?;
    let path = std::path::Path::new(image_path);
    if path.is_dir() {
	let mut result = vec!();
	println!("推理文件夹: {}", path.display());
	let entries = path.read_dir()?.filter_map(Result::ok).filter(|e| {
	    if let Some(ext) = e.path().extension().and_then(|x| x.to_str()) {
		matches!(ext.to_lowercase().as_str(),"png" | "jpg"| "jpeg")
	    } else {false}
	});
	for entry in entries {
	    let img = ImageReader::open(entry.path())?.decode()?.to_luma8();
	    let img: Vec<f32> = img.pixels().map(|p| p[0] as f32 / 255.0) .collect();
	    let input = Tensor::from_vec(img,(1,1,28,28),&Device::Cpu)?;
    
	    let pred = model.forward(&input)?.argmax(D::Minus1)?;
	    let pred = pred.to_dtype(DType::F32)?.get(0)?.to_scalar::<f32>()?;
	    result.push(pred);
	    

	}

	println!("预测结果{:?}",result);
    } else {
	let img = ImageReader::open(image_path)?.decode()?.to_luma8();
	let img: Vec<f32> = img.pixels().map(|p| p[0] as f32 / 255.0) .collect();
	let input = Tensor::from_vec(img,(1,1,28,28),&Device::Cpu)?;
	println!("开始预测");
    
	let pred = model.forward(&input)?.argmax(D::Minus1)?;
	println!("logits shape{:?} ",pred.shape());
	let pred = pred.to_dtype(DType::F32)?.get(0)?.to_scalar::<f32>()?;
    
	println!("预测结果{:?}",pred);
    }
    
    Ok(())
}
fn train(epochs: usize,batch_size: usize,learning_rate: f64) -> Result<()>{
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm,DType::F32,&Device::Cpu);


    
    let model = MnistCnn::new(vb.clone())?;
    let mut optim = AdamW::new_lr(vm.all_vars(),learning_rate)?;
    let datasets = vision::mnist::load()?;
    println!("训练图片:{:?} 训练标签:{:?} ",datasets.train_images.shape(),datasets.train_labels.shape());
    let (train_images,test_images) = (datasets.train_images.reshape((60000,1,28,28))?,
				      datasets.test_images.reshape((datasets.test_images.dims()[0],1,28,28))?);
    println!("训练图片:{:?} 测试图片:{:?}  ",train_images.shape(),test_images.shape());
    let train_images = train_images.to_dtype(DType::F32)?;
    let test_images = test_images.to_dtype(DType::F32)?;
    let train_labels = datasets.train_labels.to_dtype(DType::I64)?;
let test_labels = datasets.test_labels.to_dtype(DType::I64)?;
    let train_image_size = train_images.dim(0)?;
    println!("train image size { }", train_image_size);
    let batches_num = train_image_size / batch_size;
    let mut batch_indices = (0..batches_num).collect::<Vec<usize>>();
    
    for epoch in 0..epochs{
	let mut sum_loss = 0f32;
	
	batch_indices.shuffle(&mut thread_rng());
	for batch_index in batch_indices.iter() {
	    
	    let train_images = train_images.narrow(0,batch_index * batch_size,batch_size)?;
	    let train_labels = train_labels.narrow(0,batch_index*batch_size,batch_size)?;
	    let logits = model.forward(&train_images).expect("训练集forward失败");
	    
	    let loss = loss::cross_entropy(&logits,&train_labels)?;
	    optim.backward_step(&loss)?;
	    sum_loss += loss.to_scalar::<f32>()?;
	    
	}
	let average_loss = sum_loss / batches_num as f32;
	let test_logits = model.forward(&test_images)?;
	println!("test_logits dtype {:?} test_labels dtype {:?}",test_logits.dtype(),test_labels.dtype());
	let pred = test_logits.argmax(D::Minus1)?.to_dtype(DType::I64)?;
	
	let sum_ok = pred.eq(&test_labels)?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
	let test_acc = sum_ok as f32  / test_labels.dims1()? as f32;
	println!("{epoch:4} train loss {:8.5} test acc: {:5.2}%",average_loss,100. * test_acc);

	vm.save(format!("./model.safetensors-{}",epoch))?;
    }
    Ok(())
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
	Commands::Train { epochs, learning_rate,batch_size }  => {
	    println!("开始训练, epoch = {} , learning_rate = {}",epochs,learning_rate);
	    train(epochs.clone(),batch_size.clone(),learning_rate.clone())?;
	}
	Commands::Infer { image_path,model_path } => {
	    println!("开始推理, 模型 ={},图片={}",image_path,model_path);
	    infer(image_path,model_path)?;
	}
    }
    Ok(())    
}
