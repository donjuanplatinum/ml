use anyhow::Result;
use candle_core::{Device, Tensor,DType};
use candle_nn::{AdamW, VarBuilder, VarMap};
pub struct Rnn {
    /// 隐藏状态权重矩阵
    w_xh: Tensor,
    /// 输入权重矩阵
    w_hh: Tensor,
    /// 隐藏状态偏置
    b_h: Tensor,
    /// 全连接层权重矩阵
    w_hy: Tensor,
    /// 全连接层偏置
    b_y: Tensor,
}

#[derive(Clone)]
pub struct RnnConfig {
    /// 输入向量维度
    pub in_dim: usize,
    /// 隐藏状态维度(越大模型记忆越大但容易过拟合)
    pub hidden_dim: usize,
    /// 输出维度
    pub out_dim: usize,
    pub seq_len: usize,
}

impl Rnn {
    pub fn new(vb: &VarBuilder,config: RnnConfig) ->Result<Self>{
	let device = Device::Cpu;
	
	let w_xh = vb.get((config.in_dim, config.hidden_dim), "w_xh")?;
	let w_hh = vb.get((config.hidden_dim, config.hidden_dim), "w_hh")?;
	let b_h  = vb.get(config.hidden_dim, "b_h")?;
	let w_hy = vb.get((config.hidden_dim, config.out_dim), "w_hy")?;
	let b_y  = vb.get(config.out_dim, "b_y")?;
	Ok(Self{
	    w_xh,w_hh,b_h,w_hy,b_y
	})
    }
    /// x shape[batch,seq_len,input_dim]
    pub fn forward(&self,x: &Tensor) -> Result<(Tensor,Tensor)> {
	let (batch, seq_len, _input_dim) = {
            let shape = x.dims();
            (shape[0], shape[1], shape[2])
        };
	let hidden_dim =  self.b_h.dims()[0];
	let mut h_t = Tensor::zeros((batch,hidden_dim),DType::F32,x.device())?;
	let mut outputs = Vec::new();

	for t in 0..seq_len {
	    let x_t = x.narrow(1,t,1)?;
	    let x_t = x_t.reshape((batch,_input_dim))?;
	    let h_new = (x_t.matmul(&self.w_xh)? + h_t.matmul(&self.w_hh)? + &self.b_h)?.tanh()?;
	    let y_t = (h_new.matmul(&self.w_hy)? + &self.b_y)?;
	    h_t = h_new;
	    outputs.push(y_t);
	}
	let y = Tensor::stack(&outputs,1)?;
	Ok((y,h_t))
    }
}
fn main() -> Result<()>{
    let device = Device::Cpu;
    let vocab_size = 10000;
    let embed_dim = 128;
    let num_layers = 1;
    let num_classes = vocab_size;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap,DType::F32,&device);
    let config = RnnConfig{
	hidden_dim: 128,
	in_dim: 8,
	out_dim: 4,
	seq_len: 6,
    };
    let model = Rnn::new(&vb,config)?;
    let mut optim = AdamW::new_lr(varmap.all_vars(),0.01)?;

    for step in 0..10 {
	
    }
    
    Ok(())
}
