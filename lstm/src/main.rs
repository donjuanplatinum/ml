use anyhow::Result;
use candle_core::{Device, Tensor, DType,Module};
use candle_nn::{linear, lstm, AdamW, Linear,  Optimizer, VarBuilder, VarMap, LSTM,rnn::RNN};
use rand::Rng;
use std::collections::HashMap;
use std::fs;

fn main() -> Result<()> {
    // 1️⃣ 读取莎士比亚文本
    let text = fs::read_to_string("/home/donjuan/git/datasets/rnn/input.txt")?;
    let chars: Vec<char> = text.chars().collect();

    // 2️⃣ 建立词表
    let mut vocab: Vec<char> = chars.clone();
    vocab.sort();
    vocab.dedup();
    let vocab_size = vocab.len();
    println!("✅ Vocabulary size: {}", vocab_size);

    let stoi: HashMap<char, usize> = vocab.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let itos: HashMap<usize, char> = vocab.iter().enumerate().map(|(i, &c)| (i, c)).collect();

    // 3️⃣ 模型超参数
    let seq_len = 128;
    let batch_size = 64;
    let hidden_size = 256;
    let epochs = 10;

    // 4️⃣ Candle 模型定义
    let device = Device::Cpu; // 如果有 GPU 改成 Device::Cuda(0)
    let mut vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
    let lstm = lstm(vocab_size, hidden_size, Default::default(),vb.pp("lstm"))?;
    let linear = linear(hidden_size, vocab_size, vb.pp("linear"))?;
    let mut opt = AdamW::new_lr(vars.all_vars(), 1e-3)?;

    // 5️⃣ 训练循环
    let mut rng = rand::thread_rng();
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let steps = 2000; // 每轮抽样次数（越多训练越充分）

        for step in 0..steps {
            // 随机采样 batch
            let mut x_batch = Vec::new();
            let mut y_batch = Vec::new();

            for _ in 0..batch_size {
                let i = rng.gen_range(0..chars.len() - seq_len - 1);
                let input_seq: Vec<f32> = chars[i..i + seq_len]
                    .iter()
                    .map(|c| stoi[c] as f32)
                    .collect();
                let target = stoi[&chars[i + seq_len]] as f32;
                x_batch.push(input_seq);
                y_batch.push(target);
            }

	    let mut flat = Vec::with_capacity(batch_size * seq_len * vocab_size);

for seq in &x_batch {
    for &idx in seq {
        let mut one_hot = vec![0.0f32; vocab_size];
        one_hot[idx as usize] = 1.0;
        flat.extend(one_hot);
    }
}

let x = Tensor::new(flat, &device)?
    .reshape((batch_size, seq_len, vocab_size))?;

            let y = Tensor::new(y_batch, &device)?;

	    // batch_size 从 x 里取
let batch_size = x.dim(0)?;
let state0 = lstm.zero_state(batch_size)?;

// 执行 LSTM
let states = lstm.seq_init(&x, &state0)?;
let out = lstm.states_to_tensor(&states)?;   // [seq_len, batch_size, hidden_size]

// 取最后时间步
let last_out = out.narrow(0, seq_len - 1, 1)?.squeeze(0)?; // [batch_size, hidden_size]

// linear + loss
let logits = linear.forward(&last_out)?;
let loss = candle_nn::loss::cross_entropy(&logits.flatten_all()?, &y.flatten_all()?)?;


            // 反向传播
            opt.backward_step(&loss)?;
            total_loss += loss.to_scalar::<f32>()?;
        }

        println!("Epoch {epoch}: avg loss = {:.4}", total_loss / steps as f32);
    }

    // // 6️⃣ 文本生成（从某个字符开始）
    // let start_char = 'T';
    // let mut input = vec![stoi[&start_char] as f32];
    // let mut hidden = None;
    // let mut cell = None;

    // print!("📝 Generated text: {start_char}");
    // for _ in 0..500 {
    //     let x = Tensor::new(&input, &device)?.reshape((1, 1, 1))?;
    //     let (out, h, c) = lstm.forward(&x)?;
    //     hidden = Some(h);
    //     cell = Some(c);
    //     let logits = linear.forward(&out)?;
    //     let probs = candle_nn::ops::softmax(&logits.squeeze(0)?, DType::F32)?;
    //     let p = probs.to_vec1()?;
    //     let next_idx = sample_from_distribution(&p);
    //     let next_char = itos[&next_idx];
    //     print!("{}", next_char);
    //     input = vec![next_idx as f32];
    // }

    Ok(())
}

// 随机采样函数（根据概率分布挑选下一个字符）
fn sample_from_distribution(p: &[f32]) -> usize {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut cum = 0.0;
    let r: f32 = rng.r#gen();
    for (i, &v) in p.iter().enumerate() {
        cum += v;
        if r < cum {
            return i;
        }
    }
    p.len() - 1
}
