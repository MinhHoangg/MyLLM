# Model Configuration Guide

## 🤖 Supported Models

This chatbot uses dual-model system with automatic fallback between high-performance and balanced modes.

---

## Primary Model: dnotitia/DNA-2.0-8B

**Status:** 🔒 Gated (Requires Approval)

| Property | Value |
|----------|-------|
| **Parameters** | 8B |
| **Size** | ~16 GB |
| **Access** | Requires Hugging Face approval |
| **Best For** | Highest quality responses |

### How to Get Access:

1. **Create Hugging Face Account**
   - Visit: https://huggingface.co/join
   - Sign up with email

2. **Request Model Access**
   - Go to: https://huggingface.co/dnotitia/DNA-2.0-8B
   - Click **"Request Access"** button
   - Fill out the form
   - **Wait for approval email** (usually 1-3 days)

3. **Generate Access Token**
   - After approval, go to: https://huggingface.co/settings/tokens
   - Click **"New Token"**
   - Name it (e.g., "ChatBot Access")
   - Select **"Read"** permission
   - Copy the token

4. **Use Token in App**
   ```bash
   # Option 1: Environment variable
   export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
   streamlit run ui/app.py
   
   # Option 2: Pass to model
   # Edit ui/app.py and add hf_token parameter
   ```

---

## Fallback Models: LGAI-EXAONE Series

**Status:** ✅ Open Access (No Approval Needed)

### High-Performance Mode: EXAONE-3.5-7.8B-Instruct

| Property | Value |
|----------|-------|
| **Parameters** | 7.8B |
| **Size** | ~15.6 GB |
| **VRAM** | ~16 GB |
| **Load Time** | ~2-3 minutes |
| **Access** | Freely available |
| **Best For** | High-quality responses, complex tasks |

### Balanced Mode: EXAONE-3.5-2.4B-Instruct (Default)

| Property | Value |
|----------|-------|
| **Parameters** | 2.4B |
| **Size** | ~4.8 GB |
| **VRAM** | ~6 GB |
| **Load Time** | ~40-50 seconds |
| **Access** | Freely available |
| **Best For** | Fast responses, balanced quality |

### Why EXAONE?

✅ **No waiting** - Works immediately  
✅ **No approval** - Open access  
✅ **Two modes** - Choose speed vs quality  
✅ **Instruction-tuned** - Better at following instructions  
✅ **Multilingual** - Supports multiple languages  
✅ **Switchable** - Toggle between 2.4B and 7.8B in UI  

---

## Automatic Fallback System

The chatbot uses smart fallback logic:

```
┌─────────────────────────────────────────┐
│ 1. Try: dnotitia/DNA-2.0-8B            │
│    (Primary model)                      │
└──────────────┬──────────────────────────┘
               │
               ▼
      ┌────────────────┐
      │ Success?       │
      └────┬──────┬────┘
           │      │
          Yes     No (401/403 error)
           │      │
           │      ▼
           │  ┌─────────────────────────────────────────┐
           │  │ 2. Try: EXAONE-3.5-2.4B (Default)      │
           │  │    or EXAONE-3.5-7.8B (High Perf)      │
           │  │    (Fallback - no auth needed)         │
           │  └──────────────┬──────────────────────────┘
           │                 │
           │                 ▼
           │         ┌────────────────┐
           │         │ Success?       │
           │         └────┬──────┬────┘
           │              │      │
           │             Yes     No
           │              │      │
           ▼              ▼      ▼
    ┌──────────────────────────────────┐
    │ ✅ Model Loaded Successfully!    │
    └──────────────────────────────────┘
                      │
                      ▼
            ❌ Error: No model available
```

---

## First Run Behavior

### Scenario 1: No Approval Yet (Default - Balanced Mode)

```
🔄 Loading model dnotitia/DNA-2.0-8B on mps...
❌ Failed to load dnotitia/DNA-2.0-8B: 401 Client Error
⚠️ dnotitia/DNA-2.0-8B requires approval or authentication
→ Trying fallback model...
🔄 Attempting to load: LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
📥 Downloading model... (this may take 5-10 minutes)
✅ Successfully loaded: EXAONE-3.5-2.4B-Instruct

ℹ️ Using fallback model (Primary model pending approval)
```

**Time:** 5-10 minutes (first download only)  
**Next run:** ~40-50 seconds (model cached)

### Scenario 2: High-Performance Mode

```
🔄 Attempting to load: LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
📥 Downloading model... (this may take 15-30 minutes)
✅ Successfully loaded: EXAONE-3.5-7.8B-Instruct
```

**Time:** 15-30 minutes (first download only)  
**Next run:** ~2-3 minutes (model cached)

### Scenario 3: After Approval + Token Provided

```
🔄 Loading model dnotitia/DNA-2.0-8B on mps...
🔑 Using provided HF token for authentication
📥 Downloading model... (this may take 15-25 minutes)
✅ Successfully loaded: dnotitia/DNA-2.0-8B
```

**Time:** 15-25 minutes (first download only)  
**Next run:** ~90 seconds (model cached)

---

## Model Comparison

| Feature | DNA-2.0-8B | EXAONE-3.5-2.4B | EXAONE-3.5-7.8B |
|---------|-----------|-----------------|-----------------|
| **Access** | 🔒 Gated | ✅ Open | ✅ Open |
| **Size** | 16 GB | 4.8 GB | 15.6 GB |
| **Parameters** | 8B | 2.4B | 7.8B |
| **Speed** | ⚡⚡ Medium | ⚡⚡⚡ Fast | ⚡⚡ Medium |
| **Quality** | ⭐⭐⭐⭐⭐ Best | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Better |
| **RAM (CPU)** | 32 GB min | 8 GB min | 24 GB min |
| **VRAM (GPU)** | 20 GB min | 6 GB min | 16 GB min |
| **First Load** | 15-25 min | 5-10 min | 15-30 min |
| **Cached Load** | ~90s | ~40-50s | ~2-3 min |
| **Max Tokens** | 4096 | 4096 | 4096 |

---

## System Requirements

### For EXAONE-3.5-2.4B-Instruct (Default - Balanced Mode)

**Minimum:**
- RAM: 8 GB
- GPU VRAM: 6 GB (optional, works on CPU)
- Storage: 10 GB free

**Recommended:**
- RAM: 16 GB
- GPU: NVIDIA/AMD with 8 GB VRAM, or Apple Silicon M1/M2/M3
- Storage: 15 GB free (SSD)

### For EXAONE-3.5-7.8B-Instruct (High-Performance Mode)

**Minimum:**
- RAM: 24 GB
- GPU VRAM: 16 GB (optional, works on CPU)
- Storage: 25 GB free

**Recommended:**
- RAM: 32 GB
- GPU: NVIDIA with 24 GB VRAM, or Apple Silicon M2 Pro/Max/Ultra
- Storage: 40 GB free (SSD)

### For dnotitia/DNA-2.0-8B (Primary Model)

**Minimum:**
- RAM: 32 GB
- GPU VRAM: 20 GB (optional)
- Storage: 30 GB free

**Recommended:**
- RAM: 64 GB
- GPU: NVIDIA with 24 GB VRAM
- Storage: 50 GB free (SSD)

---

## Troubleshooting

### ❌ Error: "401 Client Error"

**Cause:** Primary model requires approval  
**Solution:** ✅ Automatic! App will use SOLAR fallback  
**To use primary:** Request access (see instructions above)

### ❌ Error: "CUDA/MPS out of memory"

**Cause:** GPU doesn't have enough VRAM  
**Solutions:**
1. Let it fallback to CPU (slower but works)
2. Switch to smaller model (2.4B instead of 7.8B)
3. Uncheck "Use High-Performance Model" in sidebar
4. Close other GPU-intensive applications

### ❌ Error: "Failed to load any model"

**Cause:** Network issue or corrupted download  
**Solutions:**
1. Check internet connection
2. Clear HuggingFace cache:
   ```bash
   # macOS/Linux
   rm -rf ~/.cache/huggingface/hub/
   
   # Windows
   rmdir /s %USERPROFILE%\.cache\huggingface
   ```
3. Restart the app

### ⚠️ Warning: "Using fallback model"

**Cause:** Primary model not accessible yet  
**Not an error!** The app is working correctly with EXAONE  
**Action:** None needed, or request access to primary model

### 🔄 Switching Between Models

Use the sidebar in the app:
1. Check "Use High-Performance Model" → 7.8B
2. Uncheck → 2.4B (default)
3. Click "🔄 Reload Model Now" to apply changes

---

## Manual Model Selection

### Via UI (Recommended)

Use the sidebar checkbox:
- **Unchecked** (default): 2.4B balanced mode
- **Checked**: 7.8B high-performance mode
- Click "🔄 Reload Model Now" to apply

### Via Code

If you want to force a specific model:

### Edit `ui/app.py`:

```python
@st.cache_resource(show_spinner=False)
def load_model(_high_parameter: bool = False):
    """Load the chatbot model with caching"""
    with st.spinner("Loading model..."):
        model = DNotitiaModel(
            high_parameter=True,  # True = 7.8B, False = 2.4B
            hf_token=None  # Or your token for primary model
        )
        return model
```

---

## Performance Tips

### 1. First Download Optimization

The app automatically downloads and caches models on first use.

For offline use, download models beforehand:

```python
# Create scripts/download_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"  # or 7.8B variant
print(f"Downloading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)

print("✅ Model downloaded successfully!")
```

Run once:
```bash
python scripts/download_model.py
```

### 2. GPU Acceleration

The app automatically uses GPU when available:
- **NVIDIA GPU**: CUDA support (recommended for 7.8B)
- **Apple Silicon (M1/M2/M3)**: MPS support (works great!)
- **AMD GPU**: ROCm support (Linux)
- **No GPU**: CPU (slower but functional)

### 3. Response Length

Default: **4096 tokens** (~3000 words) - effectively unlimited for most use cases

To adjust if needed:
```python
# In app/chatbot.py, _manual_rag method
answer = self.model.generate(prompt, max_new_tokens=2048)  # Reduce if needed
```

---

## Cache Locations

Models are cached to avoid re-downloading:

| OS | Cache Location |
|----|----------------|
| **Linux/Mac** | `~/.cache/huggingface/hub/` |
| **Windows** | `C:\Users\<username>\.cache\huggingface\hub\` |

**Disk space check:**
```bash
# Linux/Mac
du -sh ~/.cache/huggingface/

# Windows (PowerShell)
Get-ChildItem -Path $env:USERPROFILE\.cache\huggingface\ -Recurse | Measure-Object -Property Length -Sum
```

---

## FAQ

**Q: Do I need to approve the dnotitia model?**  
A: No! The app works great with EXAONE fallback. Approval is optional for using the primary model.

**Q: Which model should I use?**  
A: Start with 2.4B (default). Use 7.8B if you need higher quality and have enough VRAM.

**Q: How long does first download take?**  
A: 2.4B: 5-10 minutes | 7.8B: 15-30 minutes | DNA-2.0-8B: 15-25 minutes. Only happens once!

**Q: Can I switch between models?**  
A: Yes! Use the checkbox in sidebar and click "🔄 Reload Model Now". Both models are cached after download.

**Q: Will this work on my laptop?**  
A: Yes! 2.4B works on most laptops with 8GB RAM. 7.8B needs 16GB+ RAM or good GPU.

**Q: What's the difference between 2.4B and 7.8B?**  
A: 2.4B is faster and uses less memory. 7.8B gives better quality responses but is slower.

**Q: Can I run this offline?**  
A: Yes! After first download, models are cached locally. No internet needed after that.

**Q: Does it work on Apple Silicon (M1/M2/M3)?**  
A: Yes! MPS acceleration works great. 2.4B runs smoothly, 7.8B needs M2 Pro or better.

**Q: What's the maximum response length?**  
A: 4096 tokens (~3000 words) - effectively unlimited for most questions.

---

## Summary

✅ **App works immediately** with EXAONE-3.5-2.4B fallback  
✅ **No approval needed** to start using  
✅ **Two modes** - Balanced (2.4B) or High-Performance (7.8B)  
✅ **Switchable in UI** - Toggle between models anytime  
✅ **Automatic fallback** handles everything  
✅ **Unlimited responses** - Up to 4096 tokens (~3000 words)  
✅ **One-time download** then cached forever  
✅ **Works offline** after initial download  

**You can start chatting right away!** 🚀

The dnotitia DNA-2.0-8B model is optional - request access only if you specifically need it.
