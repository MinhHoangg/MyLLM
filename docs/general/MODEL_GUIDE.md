# Model Configuration Guide

## 🤖 Supported Models

This chatbot automatically handles model access with intelligent fallback.

---

## Primary Model: dnotitia/DNA-2.0-1.7B

**Status:** 🔒 Gated (Requires Approval)

| Property | Value |
|----------|-------|
| **Parameters** | 1.7B |
| **Size** | ~3.5 GB |
| **Access** | Requires Hugging Face approval |
| **Best For** | Lightweight, fast inference |

### How to Get Access:

1. **Create Hugging Face Account**
   - Visit: https://huggingface.co/join
   - Sign up with email

2. **Request Model Access**
   - Go to: https://huggingface.co/dnotitia/DNA-2.0-1.7B
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

## Fallback Model: upstage/SOLAR-10.7B-v1.0

**Status:** ✅ Open Access (No Approval Needed)

| Property | Value |
|----------|-------|
| **Parameters** | 10.7B |
| **Size** | ~21 GB |
| **Access** | Freely available |
| **Best For** | Better quality, more powerful |

### Why SOLAR?

✅ **No waiting** - Works immediately  
✅ **No approval** - Open access  
✅ **Better quality** - Larger model (10.7B vs 1.7B)  
✅ **Well-tested** - Popular community model  
✅ **Good performance** - Competitive with much larger models  

---

## Automatic Fallback System

The chatbot uses smart fallback logic:

```
┌─────────────────────────────────────────┐
│ 1. Try: dnotitia/DNA-2.0-1.7B          │
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
           │  ┌─────────────────────────────────────┐
           │  │ 2. Try: upstage/SOLAR-10.7B-v1.0   │
           │  │    (Fallback - no auth needed)     │
           │  └──────────────┬──────────────────────┘
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

### Scenario 1: No Approval Yet (Default)

```
🔄 Loading model dnotitia/DNA-2.0-1.7B on cuda...
❌ Failed to load dnotitia/DNA-2.0-1.7B: 401 Client Error
⚠️ dnotitia/DNA-2.0-1.7B requires approval or authentication
→ Trying fallback model...
🔄 Attempting to load: upstage/SOLAR-10.7B-v1.0
📥 Downloading model... (this may take 10-15 minutes)
✅ Successfully loaded: upstage/SOLAR-10.7B-v1.0

ℹ️ Using fallback model (Primary model pending approval)
```

**Time:** 10-15 minutes (first download only)  
**Next run:** ~30 seconds (model cached)

### Scenario 2: After Approval + Token Provided

```
🔄 Loading model dnotitia/DNA-2.0-1.7B on cuda...
🔑 Using provided HF token for authentication
📥 Downloading model... (this may take 5-10 minutes)
✅ Successfully loaded: dnotitia/DNA-2.0-1.7B
```

**Time:** 5-10 minutes (first download only)  
**Next run:** ~10 seconds (model cached)

---

## Model Comparison

| Feature | dnotitia/DNA-2.0-1.7B | upstage/SOLAR-10.7B-v1.0 |
|---------|----------------------|--------------------------|
| **Access** | 🔒 Gated | ✅ Open |
| **Size** | 3.5 GB | 21 GB |
| **Parameters** | 1.7B | 10.7B |
| **Speed** | ⚡⚡⚡ Fast | ⚡⚡ Medium |
| **Quality** | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Better |
| **RAM (CPU)** | 8 GB min | 16 GB min |
| **VRAM (GPU)** | 4 GB min | 12 GB min |
| **First Load** | 5-10 min | 10-15 min |

---

## System Requirements

### For dnotitia/DNA-2.0-1.7B

**Minimum:**
- RAM: 8 GB
- GPU VRAM: 4 GB (optional)
- Storage: 5 GB free

**Recommended:**
- RAM: 16 GB
- GPU: NVIDIA with 6 GB VRAM
- Storage: 10 GB free (SSD)

### For upstage/SOLAR-10.7B-v1.0

**Minimum:**
- RAM: 16 GB
- GPU VRAM: 12 GB (optional)
- Storage: 25 GB free

**Recommended:**
- RAM: 32 GB
- GPU: NVIDIA with 24 GB VRAM
- Storage: 40 GB free (SSD)

---

## Troubleshooting

### ❌ Error: "401 Client Error"

**Cause:** Primary model requires approval  
**Solution:** ✅ Automatic! App will use SOLAR fallback  
**To use primary:** Request access (see instructions above)

### ❌ Error: "CUDA out of memory"

**Cause:** GPU doesn't have enough VRAM  
**Solutions:**
1. Let it fallback to CPU (slower but works)
2. Use smaller model (dnotitia instead of SOLAR)
3. Upgrade GPU RAM

### ❌ Error: "Failed to load any model"

**Cause:** Network issue or corrupted download  
**Solutions:**
1. Check internet connection
2. Clear HuggingFace cache:
   ```bash
   rm -rf ~/.cache/huggingface/hub/
   ```
3. Restart the app

### ⚠️ Warning: "Using fallback model"

**Cause:** Primary model not accessible yet  
**Not an error!** The app is working correctly with SOLAR  
**Action:** None needed, or request access to primary model

---

## Manual Model Selection

If you want to force a specific model:

### Edit `ui/app.py`:

```python
@st.cache_resource
def load_model():
    """Load and cache the language model."""
    try:
        with st.spinner("Loading model..."):
            model = DNotitiaModel(
                model_name="upstage/SOLAR-10.7B-v1.0",  # Force SOLAR
                use_fallback=False,  # Disable fallback
                hf_token=None  # Or your token
            )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
```

---

## Performance Tips

### 1. First Download Optimization

Download model before first use:

```python
# Create scripts/download_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "upstage/SOLAR-10.7B-v1.0"
print(f"Downloading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("✅ Model downloaded successfully!")
```

Run once:
```bash
python scripts/download_model.py
```

### 2. Use GPU if Available

The app automatically uses GPU when available:
- **NVIDIA GPU**: CUDA support
- **Mac M1/M2/M3**: MPS support
- **No GPU**: CPU (slower but works)

### 3. Adjust Generation Settings

For faster responses, edit `ui/app.py`:

```python
model = DNotitiaModel(
    max_length=512,      # Reduce from 2048
    temperature=0.7,     # Lower = more focused
)
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
A: No! The app works great with SOLAR fallback. Approval is optional for using the primary model.

**Q: Which model is better?**  
A: SOLAR (10.7B) is more powerful but larger. Use it if you have the RAM/VRAM.

**Q: How long does first download take?**  
A: 10-15 minutes for SOLAR on good internet. Only happens once!

**Q: Can I use both models?**  
A: Yes, but only one at a time. Both are cached, so switching is fast after initial download.

**Q: Will this work on my laptop?**  
A: Yes! With 16GB RAM, SOLAR runs fine on CPU. It's just slower than GPU.

**Q: Can I add other models?**  
A: Yes! Edit `MODEL_CONFIGS` in `models/dnotitia_model.py`.

---

## Summary

✅ **App works immediately** with SOLAR fallback  
✅ **No approval needed** to start using  
✅ **Automatic fallback** handles everything  
✅ **Better quality** with SOLAR (10.7B vs 1.7B)  
✅ **One-time download** then cached forever  

**You can start chatting right away!** 🚀

The dnotitia model is optional - request access only if you specifically need it.
