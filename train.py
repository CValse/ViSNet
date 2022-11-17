import sys
import os
import argparse
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from transformer_m_visnet.module import LNNP
from transformer_m_visnet.data import DataModule
from transformer_m_visnet.utils import LoadFromFile, LoadFromCheckpoint, save_argparse, CustomApexMixedPrecisionPlugin
import torch


def get_args():
    
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--load-model', action=LoadFromCheckpoint, help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    
    # training settings
    parser.add_argument('--total-num-updates', default=1500000, type=int, help='Total number of training steps')
    parser.add_argument('--warmup-updates', default=150000, type=int, help='Number of warmup steps')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--adam-beta1', default=0.9, type=float, help='beta1 for adam')
    parser.add_argument('--adam-beta2', default=0.999, type=float, help='beta2 for adam')
    parser.add_argument('--adam-eps', default=1e-8, type=float, help='epsilon for adam')
    parser.add_argument('--end-learning-rate', default=1e-9, type=float, help='Minimum learning rate')
    parser.add_argument('--power', default=1, type=int, help='Power of polynomial decay')
    parser.add_argument('--clip-norm', default=5.0, type=float, help='gradient clipping')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay strength')
    
    # dataset specific
    parser.add_argument('--dataset-root', default='./data', type=str, help='Data storage directory')
    parser.add_argument('--AddHs', action=argparse.BooleanOptionalAction, default=False, help='Whether to add hydrogens to the molecules')
    
    # dataloader specific
    parser.add_argument('--reload', type=int, default=1, help='Reload dataloaders every n epoch')
    parser.add_argument('--standardize', action=argparse.BooleanOptionalAction, default=False, help='If true, multiply prediction by dataset std and add mean')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--inference-batch-size', default=None, type=int, help='Batchsize for validation and tests.')
    parser.add_argument('--num-workers', type=int, default=16, help='Number of workers for data prefetch')
    parser.add_argument('--update-freq', type=int, default=1, help='Number of gradient accumulation steps')

    # architectural args
    parser.add_argument('--num-atoms', default=512, type=int, help='Maximum number of atoms in a molecule')
    parser.add_argument('--num-in-degree', default=512, type=int, help='Maximum number of in degree')
    parser.add_argument('--num-out-degree', default=512, type=int, help='Maximum number of out degree')
    parser.add_argument('--num-edges', default=512, type=int, help='Maximum number of edges')
    parser.add_argument('--num-spatial', default=512, type=int, help='Number of spatial dimensions')
    parser.add_argument('--num-edge-dis', default=128, type=int, help='Maximum number of edge distances')
    parser.add_argument('--multi-hop-max-dist', default=5, type=int, help='Maximum distance for multi-hop')
    parser.add_argument('--encoder-layers', default=12, type=int, help='Number of encoder layers')
    parser.add_argument('--encoder-embed-dim', default=768, type=int, help='Encoder embedding dimension')
    parser.add_argument('--encoder-ffn-embed-dim', default=768, type=int, help='Encoder embedding dimension for FFN')
    parser.add_argument('--encoder-attention-heads', default=32, type=int, help='Number of attention heads')
    parser.add_argument('--dropout', default=0.0, type=float, help='Dropout probability')
    parser.add_argument('--attention-dropout', default=0.1, type=float, help='Attention dropout probability')
    parser.add_argument('--act-dropout', default=0.0, type=float, help='Activation dropout probability')
    parser.add_argument('--sandwich-ln', default=False, type=bool, help='Sandwich layer normalization')
    parser.add_argument('--droppath-prob', default=0.1, type=float, help='Droppath probability')
    parser.add_argument('--add-3d', action=argparse.BooleanOptionalAction, default=True, help='Whether to add 3D coordinates')
    parser.add_argument('--num-3d-bias-kernel', default=128, type=int, help='Number of 3D bias and kernel')
    parser.add_argument('--no-2d', action=argparse.BooleanOptionalAction, default=False, help='Whether to remove 2D topology')
    parser.add_argument('--noise-scale', default=0.2, type=float, help='noise scale')
    parser.add_argument('--mode-prob', type=str, default='0.2,0.2,0.6', help='Probability of each mode {2D+3D, 2D, 3D}')

    # other args
    parser.add_argument('--ngpus', type=int, default=-1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--precision', type=int, default=16, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--amp-backend', type=str, default='native', choices=['native', 'apex'], help='Automatic mixed precision backend')
    parser.add_argument('--log-dir', '-l', default='./logs', help='log file')
    parser.add_argument('--save-interval', type=int, default=1, help='Save interval, one save per n epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--distributed-backend', default='ddp', help='Distributed backend: dp, ddp, ddp2')
    parser.add_argument('--accelerator', default='gpu', help='Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "auto")')
    parser.add_argument('--redirect', action=argparse.BooleanOptionalAction, default=False, help='Redirect stdout and stderr to log_dir/log')
    parser.add_argument('--task', type=str, default='train', help='Task to perform: train, inference')
    parser.add_argument('--inference-dataset', type=str, default='valid', help='Dataset to perform inference on: valid, test-dev, test-challenge')

    args = parser.parse_args()

    if args.redirect:
        os.makedirs(args.log_dir, exist_ok=True)
        sys.stdout = open(os.path.join(args.log_dir, "log"), "w")
        sys.stderr = sys.stdout
        logging.getLogger("pytorch_lightning").addHandler(logging.StreamHandler(sys.stdout))

    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size
        
    if args.task == 'train':
        save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])
    return args

def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)

    # initialize data module
    data = DataModule(args)
    data.prepare_data()
    data.split_compute()

    default = ",".join(str(i) for i in range(torch.cuda.device_count()))
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
    
    dir_name = f"ngpus_{len(cuda_visible_devices)}_bs_{args.batch_size * args.update_freq}" + \
               f"_tsteps_{args.total_num_updates}_wsteps_{args.warmup_updates}" + \
               f"_seed_{args.seed}" + \
               f"_L{args.encoder_layers}_D{args.encoder_embed_dim}_F{args.encoder_ffn_embed_dim}_H{args.encoder_attention_heads}" + \
               f"_lr_{args.lr}" + \
               f"_clip_{args.clip_norm}_dp_{args.dropout}_attn_dp_{args.attention_dropout}_wd_{args.weight_decay}_dpp_{args.droppath_prob}_noisescale_{args.noise_scale}_mode_prob_{args.mode_prob}"
    
    if args.load_model is None:

        args.log_dir = os.path.join(args.log_dir, dir_name)
        if os.path.exists(args.log_dir):
            if os.path.exists(os.path.join(args.log_dir, "last.ckpt")):
                args.load_model = os.path.join(args.log_dir, "last.ckpt")
            csv_path = os.path.join(args.log_dir, "metrics.csv")
            while os.path.exists(csv_path):
                csv_path = csv_path + '.bak'
            if os.path.exists(os.path.join(args.log_dir, "metrics.csv")):
                os.rename(os.path.join(args.log_dir, "metrics.csv"), csv_path)

    # initialize lightning module
    model = LNNP(args, mean=data.mean, std=data.std)
    print('Model Parameters:', sum(p.numel() for p in model.parameters()))
    
    if args.task == "train":
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.log_dir,
            monitor="val_epoch_loss",
            save_top_k=10,
            save_last=True,
            every_n_epochs=args.save_interval,
            filename="{epoch}-{val_epoch_loss:.4f}",
        )

        tb_logger = TensorBoardLogger(args.log_dir, name="tensorbord", version="", default_hp_metric=False)
        csv_logger = CSVLogger(args.log_dir, name="", version="")
        ddp_plugin = DDPPlugin(find_unused_parameters=False)
        
        if args.amp_backend == "apex":
            apex_plugin = CustomApexMixedPrecisionPlugin(amp_level="O1", max_loss_scale=128, min_loss_scale=0.0001)
        else:
            apex_plugin = None
        
        trainer = pl.Trainer(
            max_steps=args.total_num_updates,
            gpus=args.ngpus,
            num_nodes=args.num_nodes,
            accelerator=args.accelerator,
            default_root_dir=args.log_dir,
            auto_lr_find=False,
            callbacks=[checkpoint_callback],
            logger=[tb_logger, csv_logger],
            reload_dataloaders_every_n_epochs=args.reload,
            precision=args.precision,
            strategy=ddp_plugin,
            detect_anomaly=False,
            enable_progress_bar=True,
            plugins=apex_plugin,
            gradient_clip_algorithm='norm',
            gradient_clip_val=args.clip_norm,
            accumulate_grad_batches=args.update_freq,
        )
        
        trainer.fit(model, datamodule=data, ckpt_path=args.load_model)
        
    if args.task == 'inference':
        
        test_trainer = pl.Trainer(
            logger=False,
            max_epochs=-1,
            num_nodes=1,
            gpus=1,
            default_root_dir=args.log_dir,
            enable_progress_bar=True,
        )
        test_trainer.test(model=model, ckpt_path=args.load_model, datamodule=data)
        
        from ogb.lsc import PCQM4Mv2Evaluator
        
        evaluator = PCQM4Mv2Evaluator()
        
        if args.inference_dataset == 'valid':
            valid_mae = evaluator.eval({'y_true': model.inference_results['y_true'].cpu().numpy(), 'y_pred': model.inference_results['y_pred'].cpu().numpy()})
            print('The Valid MAE is:', valid_mae['mae'])
        elif args.inference_dataset == 'test-dev':
            evaluator.save_test_submission(input_dict={'y_pred': model.inference_results['y_pred'].cpu().numpy()}, dir_path=args.log_dir, mode='test-dev')
        else:
            evaluator.save_test_submission(input_dict={'y_pred': model.inference_results['y_pred'].cpu().numpy()}, dir_path=args.log_dir, mode='test-challenge')

if __name__ == "__main__":
    main()