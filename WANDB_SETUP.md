# Weights & Biases Setup

This project uses W&B for experiment tracking. Follow these steps to get set up:

## 1. Get Your W&B API Key

1. Go to https://wandb.ai/
2. Sign up or log in
3. Go to your account settings: https://wandb.ai/settings
4. Scroll down to the "API keys" section
5. Copy your API key

## 2. Configure Your .env File

Open the `.env` file in this directory and fill in your credentials:

```bash
# Required: Your W&B API key
WANDB_API_KEY=your_actual_api_key_here

# Required: Your W&B username or team name
WANDB_ENTITY=your_wandb_username

# Optional: Project name (default is "maslow-rl")
WANDB_PROJECT=maslow-rl
```

**Important:**
- Replace `your_actual_api_key_here` with your actual API key from step 1
- Replace `your_wandb_username` with your W&B username (found at https://wandb.ai/settings)
- The `.env` file is in `.gitignore` and will NOT be committed to git

## 3. Test Your Setup

Run this command to verify W&B is configured:

```bash
source venv/bin/activate
python -c "import wandb; wandb.login()"
```

If it says "W&B is configured properly", you're all set!

## 4. Run Training with W&B

Both training configs are already set up to log to W&B:

```bash
# Gated run
python train.py --config config.json

# Linear run
python train.py --config config_linear.json
```

You'll see your runs appear at: https://wandb.ai/YOUR_USERNAME/maslow-rl

## Offline Mode (Optional)

If you want to run experiments without internet or save logs locally first:

Uncomment this line in `.env`:
```bash
WANDB_MODE=offline
```

Then sync later with:
```bash
wandb sync wandb/offline-run-*
```

## What Gets Logged

W&B will track:
- `reward_total_mean` - Total reward across batch
- `r_a_mean` - Structure reward (Tier A)
- `r_b_mean` - Correctness reward (Tier B)
- `gate_b_mean` - Gating value (gated run only)
- `accuracy` - Answer accuracy
- `json_parse_rate` - Valid JSON rate
- All hyperparameters from config files
- Model checkpoints (if enabled)

This lets you compare learning curves between linear and gated runs!
