---
tags:
- masked-auto-encoding
- generated_from_trainer
datasets:
- imagefolder
model-index:
- name: MAE_full100_4
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# MAE_full100_4

This model is a fine-tuned version of [](https://huggingface.co/) on the /home/NUS-NeuralNetworks dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4007

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 160
- eval_batch_size: 60
- seed: 12
- gradient_accumulation_steps: 4
- total_train_batch_size: 640
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 100.0

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.9465        | 0.98  | 31   | 0.7775          |
| 0.7315        | 2.0   | 63   | 0.7390          |
| 0.7357        | 2.98  | 94   | 0.7146          |
| 0.6951        | 4.0   | 126  | 0.7036          |
| 0.7104        | 4.98  | 157  | 0.7020          |
| 0.6875        | 6.0   | 189  | 0.7011          |
| 0.7091        | 6.98  | 220  | 0.7006          |
| 0.6861        | 8.0   | 252  | 0.7017          |
| 0.7083        | 8.98  | 283  | 0.7001          |
| 0.6852        | 10.0  | 315  | 0.6974          |
| 0.7069        | 10.98 | 346  | 0.7050          |
| 0.6841        | 12.0  | 378  | 0.6939          |
| 0.7017        | 12.98 | 409  | 0.6927          |
| 0.678         | 14.0  | 441  | 0.6908          |
| 0.6971        | 14.98 | 472  | 0.6874          |
| 0.6753        | 16.0  | 504  | 0.6898          |
| 0.6943        | 16.98 | 535  | 0.6851          |
| 0.669         | 18.0  | 567  | 0.6747          |
| 0.6927        | 18.98 | 598  | 0.6728          |
| 0.6489        | 20.0  | 630  | 0.6589          |
| 0.659         | 20.98 | 661  | 0.6653          |
| 0.6295        | 22.0  | 693  | 0.6378          |
| 0.6436        | 22.98 | 724  | 0.6416          |
| 0.6064        | 24.0  | 756  | 0.6062          |
| 0.6088        | 24.98 | 787  | 0.5958          |
| 0.577         | 26.0  | 819  | 0.5815          |
| 0.5827        | 26.98 | 850  | 0.5686          |
| 0.5411        | 28.0  | 882  | 0.5672          |
| 0.5522        | 28.98 | 913  | 0.5345          |
| 0.5165        | 30.0  | 945  | 0.5207          |
| 0.5227        | 30.98 | 976  | 0.5147          |
| 0.4979        | 32.0  | 1008 | 0.5040          |
| 0.5072        | 32.98 | 1039 | 0.4980          |
| 0.485         | 34.0  | 1071 | 0.4915          |
| 0.4958        | 34.98 | 1102 | 0.4920          |
| 0.4762        | 36.0  | 1134 | 0.4839          |
| 0.4865        | 36.98 | 1165 | 0.4795          |
| 0.4687        | 38.0  | 1197 | 0.4753          |
| 0.4804        | 38.98 | 1228 | 0.4767          |
| 0.4611        | 40.0  | 1260 | 0.4688          |
| 0.4717        | 40.98 | 1291 | 0.4699          |
| 0.4536        | 42.0  | 1323 | 0.4612          |
| 0.4665        | 42.98 | 1354 | 0.4583          |
| 0.4482        | 44.0  | 1386 | 0.4589          |
| 0.4611        | 44.98 | 1417 | 0.4552          |
| 0.4424        | 46.0  | 1449 | 0.4505          |
| 0.4578        | 46.98 | 1480 | 0.4529          |
| 0.4391        | 48.0  | 1512 | 0.4478          |
| 0.4514        | 48.98 | 1543 | 0.4447          |
| 0.4356        | 50.0  | 1575 | 0.4439          |
| 0.4481        | 50.98 | 1606 | 0.4437          |
| 0.4318        | 52.0  | 1638 | 0.4419          |
| 0.444         | 52.98 | 1669 | 0.4402          |
| 0.4274        | 54.0  | 1701 | 0.4364          |
| 0.4406        | 54.98 | 1732 | 0.4363          |
| 0.4247        | 56.0  | 1764 | 0.4358          |
| 0.4375        | 56.98 | 1795 | 0.4318          |
| 0.4217        | 58.0  | 1827 | 0.4310          |
| 0.4334        | 58.98 | 1858 | 0.4299          |
| 0.4192        | 60.0  | 1890 | 0.4286          |
| 0.4308        | 60.98 | 1921 | 0.4243          |
| 0.415         | 62.0  | 1953 | 0.4244          |
| 0.4277        | 62.98 | 1984 | 0.4228          |
| 0.4131        | 64.0  | 2016 | 0.4229          |
| 0.4255        | 64.98 | 2047 | 0.4196          |
| 0.41          | 66.0  | 2079 | 0.4193          |
| 0.4228        | 66.98 | 2110 | 0.4194          |
| 0.4083        | 68.0  | 2142 | 0.4177          |
| 0.4203        | 68.98 | 2173 | 0.4165          |
| 0.4066        | 70.0  | 2205 | 0.4152          |
| 0.4185        | 70.98 | 2236 | 0.4133          |
| 0.4041        | 72.0  | 2268 | 0.4141          |
| 0.4166        | 72.98 | 2299 | 0.4121          |
| 0.4031        | 74.0  | 2331 | 0.4111          |
| 0.4142        | 74.98 | 2362 | 0.4117          |
| 0.4001        | 76.0  | 2394 | 0.4118          |
| 0.4137        | 76.98 | 2425 | 0.4080          |
| 0.3995        | 78.0  | 2457 | 0.4088          |
| 0.4117        | 78.98 | 2488 | 0.4092          |
| 0.3978        | 80.0  | 2520 | 0.4066          |
| 0.4101        | 80.98 | 2551 | 0.4061          |
| 0.3968        | 82.0  | 2583 | 0.4052          |
| 0.4092        | 82.98 | 2614 | 0.4046          |
| 0.3954        | 84.0  | 2646 | 0.4048          |
| 0.4078        | 84.98 | 2677 | 0.4043          |
| 0.3949        | 86.0  | 2709 | 0.4042          |
| 0.407         | 86.98 | 2740 | 0.4027          |
| 0.3935        | 88.0  | 2772 | 0.4028          |
| 0.406         | 88.98 | 2803 | 0.4022          |
| 0.3931        | 90.0  | 2835 | 0.4018          |
| 0.4051        | 90.98 | 2866 | 0.4015          |
| 0.3923        | 92.0  | 2898 | 0.4012          |
| 0.4047        | 92.98 | 2929 | 0.4004          |
| 0.3912        | 94.0  | 2961 | 0.4005          |
| 0.4037        | 94.98 | 2992 | 0.4000          |
| 0.3909        | 96.0  | 3024 | 0.4001          |
| 0.4034        | 96.98 | 3055 | 0.4008          |
| 0.3907        | 98.0  | 3087 | 0.4001          |
| 0.3964        | 98.41 | 3100 | 0.4001          |


### Framework versions

- Transformers 4.35.0.dev0
- Pytorch 2.0.0
- Datasets 2.12.0
- Tokenizers 0.14.1