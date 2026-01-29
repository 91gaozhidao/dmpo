

August 13, 2025
7:33

 - Lift Task weight=0.5

  Pre-training Commands

  1. ShortCut Flow Baseline
  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_mlp_img denoising_steps=20

  2. MeanFlow Baseline
  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_meanflow_mlp_img

  3. ReFlow Baseline
  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_reflow_mlp_img

  4. ShortCut + InfoNCE L2 dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_mlp_img denoising_steps=20

  5. ShortCut + InfoNCE Cosine dispersive loss

  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_cosine_mlp_img
  denoising_steps=20

  6. ShortCut + Hinge dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_hinge_mlp_img
  denoising_steps=20

  7. ShortCut + Covariance dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_shortcut_dispersive_covariance_mlp_img
  denoising_steps=20

  8. MeanFlow + dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/lift --config-name=pre_meanflow_dispersive_mlp_img

  Pre-training Evaluation Commands

  1. ShortCut Flow Baseline Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=

  2. MeanFlow Baseline Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_meanflow_mlp_img base_policy_path=

  3. ReFlow Baseline Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_reflow_mlp_img_new base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_meanflow_mlp_img base_policy_path=

  Fine-tuning Commands

  1. ShortCut Flow Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  2. MeanFlow Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  3. ReFlow Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/lift --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  Fine-tuning Evaluation Commands

  1. ShortCut Flow Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=

  2. MeanFlow Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_meanflow_mlp_img base_policy_path=

  3. ReFlow Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_reflow_mlp_img_new base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/lift --config-name=eval_meanflow_mlp_img base_policy_path=

  ---


August 13, 2025
7:34

 - Can Task weight=0.5

  Pre-training Commands

  1. ShortCut Flow Baseline
  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_mlp_img denoising_steps=20

  2. MeanFlow Baseline
  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_meanflow_mlp_img

  3. ReFlow Baseline
  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_reflow_mlp_img

  4. ShortCut + InfoNCE L2 dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_mlp_img denoising_steps=20

  5. ShortCut + InfoNCE Cosine dispersive loss

  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_cosine_mlp_img
  denoising_steps=20

  6. ShortCut + Hinge dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_hinge_mlp_img
  denoising_steps=20

  7. ShortCut + Covariance dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_shortcut_dispersive_covariance_mlp_img
  denoising_steps=20

  8. MeanFlow + dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_meanflow_dispersive_mlp_img

  Pre-training Evaluation Commands

  1. ShortCut Flow Baseline Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  2. MeanFlow Baseline Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_meanflow_mlp_img base_policy_path=

  3. ReFlow Baseline Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_reflow_mlp_img_new base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_meanflow_mlp_img base_policy_path=

  Fine-tuning Commands

  1. ShortCut Flow Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  2. MeanFlow Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  3. ReFlow Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  Fine-tuning Evaluation Commands

  1. ShortCut Flow Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  2. MeanFlow Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_meanflow_mlp_img base_policy_path=

  3. ReFlow Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_reflow_mlp_img_new base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/can --config-name=eval_meanflow_mlp_img base_policy_path=

  ---


August 13, 2025
7:33

- Square Task weight=0.5

  Pre-training Commands

  1. ShortCut Flow Baseline
  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_mlp_img denoising_steps=20

  2. MeanFlow Baseline

  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_meanflow_mlp_img

  3. ReFlow Baseline
  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_reflow_mlp_img

  4. ShortCut + InfoNCE L2 dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_mlp_img
  denoising_steps=20

  5. ShortCut + InfoNCE Cosine dispersive loss

  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_cosine_mlp_img
  denoising_steps=20

  6. ShortCut + Hinge dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_hinge_mlp_img
  denoising_steps=20

  7. ShortCut + Covariance dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_dispersive_covariance_mlp_img
  denoising_steps=20

  8. MeanFlow + dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_meanflow_dispersive_mlp_img

  Pre-training Evaluation Commands

  1. ShortCut Flow Baseline Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=

  2. MeanFlow Baseline Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_meanflow_mlp_img base_policy_path=

  3. ReFlow Baseline Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_reflow_mlp_img_new base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_meanflow_mlp_img base_policy_path=

  Fine-tuning Commands

  1. ShortCut Flow Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  2. MeanFlow Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  3. ReFlow Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/square --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  Fine-tuning Evaluation Commands

  1. ShortCut Flow Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=

  2. MeanFlow Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_meanflow_mlp_img base_policy_path=

  3. ReFlow Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_reflow_mlp_img_new base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/square --config-name=eval_meanflow_mlp_img base_policy_path=



August 13, 2025
7:34

 - Transport Task weight=0.5

  Pre-training Commands

  1. ShortCut Flow Baseline
  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_mlp_img denoising_steps=20

  2. MeanFlow Baseline

  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_meanflow_mlp_img

  3. ReFlow Baseline
  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_reflow_mlp_img

  4. ShortCut + InfoNCE L2 dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_mlp_img
  denoising_steps=20

  5. ShortCut + InfoNCE Cosine dispersive loss

  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_cosine_mlp_img
  denoising_steps=20

  6. ShortCut + Hinge dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_hinge_mlp_img
  denoising_steps=20

  7. ShortCut + Covariance dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_shortcut_dispersive_covariance_mlp_img
  denoising_steps=20

  8. MeanFlow + dispersive loss
  python script/run.py --config-dir=cfg/robomimic/pretrain/transport --config-name=pre_meanflow_dispersive_mlp_img

  Pre-training Evaluation Commands

  1. ShortCut Flow Baseline Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  2. MeanFlow Baseline Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_meanflow_mlp_img base_policy_path=

  3. ReFlow Baseline Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_reflow_mlp_img_new base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_meanflow_mlp_img base_policy_path=

  Fine-tuning Commands

  1. ShortCut Flow Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  2. MeanFlow Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  3. ReFlow Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss Fine-tuning
  python script/run.py --config-dir=cfg/robomimic/finetune/transport --config-name=ft_ppo_reflow_mlp_img base_policy_path=

  Fine-tuning Evaluation Commands

  1. ShortCut Flow Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  2. MeanFlow Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_meanflow_mlp_img base_policy_path=

  3. ReFlow Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_reflow_mlp_img_new base_policy_path=

  4. ShortCut + InfoNCE L2 dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  5. ShortCut + InfoNCE Cosine dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  6. ShortCut + Hinge dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  7. ShortCut + Covariance dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_shortcut_mlp_img base_policy_path=

  8. MeanFlow + dispersive loss Fine-tuning Evaluation
  python script/run.py --config-dir=cfg/robomimic/eval/transport --config-name=eval_meanflow_mlp_img base_policy_path=
