SHELL := /bin/bash
CONDA_ENV := adamimic
CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)
LD_PATH := LD_LIBRARY_PATH=$$CONDA_PREFIX/lib:$$LD_LIBRARY_PATH
PYTHON := python legged_gym/legged_gym/scripts
ROBOT := g1_dof27
TASKS := badminton_hit tennis_hit high_jump far_jump triple_jump jump_step_up jump_step_down
LOG_DIR := exp

TASK := badminton_hit
STAGE := stage1
# Default: show available targets
#
# Usage:
#   make stage1 TASK=badminton_hit
#   make stage1 TASK=badminton_hit RESUME=exp/g1_dof27/badminton_hit/adamimic_stage1/<timestamp>/model_<iter>.pt
#   make stage2 TASK=badminton_hit CHECKPOINT=exp/g1_dof27/badminton_hit/adamimic_stage1/<timestamp>/model_40000.pt
#   make play TASK=badminton_hit RESUME=exp/g1_dof27/badminton_hit/adamimic_stage2/<timestamp>/model_10000.pt
#   make logs TASK=badminton_hit
#   make status TASK=badminton_hit STAGE=stage1
#   make tensorboard TASK=badminton_hit
help:
	@echo "Usage: make <target> TASK=<task_name>"
	@echo ""
	@echo "Targets:"
	@echo "  stage1        Train stage 1"
	@echo "  stage1 RESUME=   Resume stage 1 from checkpoint"
	@echo "  stage2        Train stage 2 (requires CHECKPOINT=path/to/stage1_ckpt)"
	@echo "  play          Play a trained policy (requires RESUME=path/to/stage2_ckpt)"
	@echo "  logs          Tail training logs"
	@echo "  status        Show training status (STAGE=stage1 or stage2)"
	@echo "  tensorboard   Launch tensorboard"
	@echo ""
	@echo "Available tasks: $(TASKS)"
	@echo ""
	@echo "Examples:"
	@echo "  make stage1 TASK=badminton_hit"
	@echo "  make stage1 TASK=badminton_hit RESUME=exp/g1_dof27/badminton_hit/adamimic_stage1/.../model_500.pt"
	@echo "  make stage2 TASK=badminton_hit CHECKPOINT=exp/g1_dof27/badminton_hit/.../model_40000.pt"
	@echo "  make play TASK=badminton_hit RESUME=exp/g1_dof27/badminton_hit/.../model_10000.pt"
	@echo "  make logs TASK=badminton_hit"
	@echo "  make status TASK=badminton_hit STAGE=stage1"
	@echo "  make tensorboard TASK=badminton_hit"

# Train stage 1
# Usage: make stage1 TASK=badminton_hit
# Resume: make stage1 TASK=badminton_hit RESUME=exp/g1_dof27/badminton_hit/adamimic_stage1/<timestamp>/model_<iter>.pt
stage1:
	@echo ">>> Training stage 1: $(TASK)"
	$(CONDA_ACTIVATE) && $(LD_PATH) \
	$(PYTHON)/train.py +dataset=$(ROBOT)/$(TASK) +algorithm=adamimic/stage1 +robot=$(ROBOT) \
	$(if $(RESUME),resume_path=$(RESUME) algo.runner.resume=true,)

# Train stage 2 (depends on stage1 checkpoint)
# Usage: make stage2 TASK=badminton_hit CHECKPOINT=exp/g1_dof27/badminton_hit/adamimic_stage1/<timestamp>/model_40000.pt
stage2:
	@echo ">>> Training stage 2: $(TASK)"
	$(CONDA_ACTIVATE) && $(LD_PATH) \
	$(PYTHON)/train.py +dataset=$(ROBOT)/$(TASK) +algorithm=adamimic/stage2 +robot=$(ROBOT) checkpoint_path=$(CHECKPOINT)

# Play a trained policy (requires stage2 checkpoint)
# Usage: make play TASK=badminton_hit RESUME=exp/g1_dof27/badminton_hit/adamimic_stage2/<timestamp>/model_10000.pt
play:
	@echo ">>> Playing: $(TASK)"
	$(CONDA_ACTIVATE) && $(LD_PATH) \
	$(PYTHON)/play.py +dataset=$(ROBOT)/$(TASK) +algorithm=adamimic/stage2 +robot=$(ROBOT) resume_path=$(RESUME)

# Launch tensorboard
# Usage: make tensorboard TASK=badminton_hit
tensorboard:
	$(CONDA_ACTIVATE) && tensorboard --logdir=$(LOG_DIR)/$(ROBOT)/$(TASK) --bind_all --port 6006

# Show training logs (auto-detect latest run directory)
# Usage: make logs TASK=badminton_hit
logs:
	@LOG=$$(find $(LOG_DIR)/$(ROBOT)/$(TASK) -name train.log -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-) && \
	if [ -n "$$LOG" ] && [ -s "$$LOG" ]; then \
		echo ">>> Tailing: $$LOG" && tail -f $$LOG; \
	elif [ -f train_stage1.log ]; then \
		echo ">>> Tailing: train_stage1.log (current nohup session)" && tail -f train_stage1.log; \
	else \
		echo "No logs found for $(TASK)"; exit 1; fi

# Show training status with reasonable range benchmarks
# Usage: make status TASK=badminton_hit STAGE=stage1
status:
	$(CONDA_ACTIVATE) && python scripts/check_training.py /home/ubuntu22/sourcecode/exp/$(ROBOT)/$(TASK)/adamimic_$(STAGE) $(STAGE)

.PHONY: help stage1 stage2 play logs status tensorboard