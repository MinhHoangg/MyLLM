# Scripts Directory

Utility scripts for testing and maintenance.

---

## 📋 Available Scripts

### 1. `check_model_status.py`
**Purpose**: Check which model will be loaded before running the app

**Usage**:
```bash
python scripts/check_model_status.py
```

**What it does**:
- Shows available models
- Checks for HF authentication token
- Explains expected behavior
- Provides recommendations

**When to use**:
- Before first run
- After setting up HF token
- To verify configuration

---

### 2. `test_model.py`
**Purpose**: Test model loading and generation

**Usage**:
```bash
python scripts/test_model.py
```

**What it does**:
- Attempts to load model with fallback
- Shows which model loaded
- Tests text generation
- Verifies everything works

**When to use**:
- After installation
- To verify model access
- To test generation quality
- To troubleshoot issues

---

## 🔧 Quick Examples

### Check if you need model approval
```bash
python scripts/check_model_status.py
```

### Test if everything works
```bash
python scripts/test_model.py
```

### Both in sequence
```bash
python scripts/check_model_status.py && python scripts/test_model.py
```

---

## 📊 Expected Output Examples

### `check_model_status.py` - No Token

```
═══════════════════════════════════════════════════════════════════
MODEL STATUS CHECKER
═══════════════════════════════════════════════════════════════════

ℹ️  No HF token found in environment
   Open models will be used

──────────────────────────────────────────────────────────────────
Available Models:
──────────────────────────────────────────────────────────────────

1. dnotitia/DNA-2.0-1.7B
   Description: DNotitia DNA 2.0 (1.7B parameters)
   Size: 3.5 GB
   Status: 🔒 GATED (requires approval)
   → Will skip (no token provided)

2. upstage/SOLAR-10.7B-v1.0
   Description: Upstage SOLAR (10.7B parameters) - Open Access
   Size: 21.0 GB
   Status: ✅ OPEN ACCESS
   → Available for immediate use

──────────────────────────────────────────────────────────────────
Expected Behavior:
──────────────────────────────────────────────────────────────────

Without HF token:
  1. Skip dnotitia/DNA-2.0-1.7B (requires auth)
  2. Use: upstage/SOLAR-10.7B-v1.0 ✅
```

### `test_model.py` - Success with Fallback

```
═══════════════════════════════════════════════════════════════════
TESTING MODEL FALLBACK SYSTEM
═══════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────
🔄 Testing Model Load (with fallback enabled)...
──────────────────────────────────────────────────────────────────

Attempting to load primary model...
❌ Failed to load dnotitia/DNA-2.0-1.7B: 401 Client Error
⚠️ dnotitia/DNA-2.0-1.7B requires approval or authentication
→ Trying fallback model...
🔄 Attempting to load: upstage/SOLAR-10.7B-v1.0
✅ Successfully loaded: upstage/SOLAR-10.7B-v1.0

═══════════════════════════════════════════════════════════════════
✅ MODEL LOADED SUCCESSFULLY!
═══════════════════════════════════════════════════════════════════

📊 Model Information:
   Requested Model: dnotitia/DNA-2.0-1.7B
   Loaded Model: upstage/SOLAR-10.7B-v1.0
   Device: cuda
   Description: Upstage SOLAR (10.7B parameters) - Open Access
   Is Fallback: True

ℹ️  NOTE: Primary model was not accessible.
    The system automatically fell back to an open model.
    This is expected behavior if you haven't requested
    access to the dnotitia model yet.

──────────────────────────────────────────────────────────────────
🧪 Testing Generation...
──────────────────────────────────────────────────────────────────

Prompt: What is artificial intelligence?

Generating response...

Response:
Artificial intelligence (AI) refers to the simulation of human 
intelligence in machines that are programmed to think like humans 
and mimic their actions. It encompasses various technologies...

═══════════════════════════════════════════════════════════════════
✅ ALL TESTS PASSED!
═══════════════════════════════════════════════════════════════════

Your chatbot is ready to use! 🎉
```

---

## 🐛 Troubleshooting

### Script not found
```bash
# Make sure you're in the project root
cd /path/to/ChatBot

# Then run
python scripts/check_model_status.py
```

### Permission denied
```bash
# Make scripts executable
chmod +x scripts/*.py

# Then run
./scripts/check_model_status.py
```

### Import errors
```bash
# Activate virtual environment first
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Then run script
python scripts/test_model.py
```

---

## 💡 Tips

1. **Always check status before testing**: Run `check_model_status.py` first
2. **First test takes time**: Model download can take 10-15 minutes
3. **Subsequent tests are fast**: Models are cached after first download
4. **Check internet connection**: Scripts need to download models
5. **Verify disk space**: SOLAR model needs ~25GB free space

---

## 📚 Related Documentation

- [MODEL_GUIDE.md](../MODEL_GUIDE.md) - Complete model documentation
- [FALLBACK_SUMMARY.md](../FALLBACK_SUMMARY.md) - Fallback system overview
- [README.md](../README.md) - Main project documentation

---

**Happy Testing!** 🚀
