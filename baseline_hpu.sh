CUDA_VISIBLE_DEVICES=3 python run_hpunet.py                                                   \
--log_name base_hpu                                                                   \
--data_dir /mnt/epnfs/zhshen/RL_UIE/uieb/BIG_UIEB                                      \
--in_ch 3            --out_ch 3                                                         \
--intermediate_ch 16 32 64 64 64                                                       \
--scale_depth 1                                                                         \
--kernel_size 7 7 7 5 3                                                                 \
--padding_mode zeros                                                                    \
\
--latent_num 5                                                                          \
--latent_chs 1 1 1 1 1                                                                  \
--latent_locks 0 0 0 0 0                                                                \
\
--rec_type MSE                                                                          \
\
--loss_type ELBO        --beta 1.0                                                      \
\
--epochs 500            --bs 16                                                        \
--optimizer adamax                     --wd 1e-5                                        \
--lr 1e-4                              --scheduler_type cons                            \
\