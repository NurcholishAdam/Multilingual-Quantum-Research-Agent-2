# REPAIR Integration - Quick Reference

One-page reference for REPAIR model editing integration.

## 🚀 Quick Start

```bash
# Enable REPAIR
export ENABLE_REPAIR=true

# Run demo
python demo_repair_integration.py

# Run tests
python test_model_editing_integration.py
```

## 📦 Components

| Component | File | Purpose |
|-----------|------|---------|
| DualMemoryEditor | `model_editing.py` | Core REPAIR editing |
| REPAIRInferenceWrapper | `REPAIRInferenceWrapper.py` | Inference with error detection |
| HealthChecker | `generate_and_validate.py` | Response validation |
| generate_and_validate | `generate_and_validate.py` | Correction workflow |

## 💻 Basic Usage

```python
from multilingual_research_agent import MultilingualResearchAgent

# Create agent with REPAIR
agent = MultilingualResearchAgent(enable_repair=True)

# Generate with automatic correction
response = agent.generate_with_repair("Your query here")

# Get statistics
stats = agent.get_repair_statistics()
```

## 🔧 Configuration

```python
from social_science_extensions.model_editing import REPAIRConfig

config = REPAIRConfig(
    mask_ratio=0.2,           # 20% of parameters
    err_thresh=0.85,          # 85% error threshold
    distill_weight=1.0,       # Distillation weight
    pruning_max=10000,        # Max parameters to prune
    learning_rate=1e-4,       # Learning rate
    num_epochs=3              # Epochs per edit
)
```

## 📊 Metrics

| Metric | Range | Target | Description |
|--------|-------|--------|-------------|
| Reliability | 0-1 | >0.8 | Correctness of edits |
| Locality | 0-1 | >0.9 | Knowledge preservation |
| Generalization | 0-1 | >0.7 | Edit generalization |
| Success Rate | 0-1 | >0.95 | Edit success rate |

## 🔄 Workflow

```
Query → Generate → Health Check → Validate
                        ↓
                   Unhealthy?
                        ↓
                  Apply REPAIR
                        ↓
                   Regenerate
```

## 📝 Examples

### Self-Healing Generation

```python
agent = MultilingualResearchAgent(enable_repair=True)
response = agent.generate_with_repair("What is AI?")
```

### Manual Edit

```python
edits = [("Query", "Correct Answer", "Locality Prompt")]
agent.editor.apply_edits(edits)
```

### Batch Editing

```python
edits = [
    ("Q1", "A1", "L1"),
    ("Q2", "A2", "L2"),
    ("Q3", "A3", "L3")
]
agent.editor.apply_edits(edits)
```

## 🧪 Testing

```bash
# Run all tests
python test_model_editing_integration.py

# Run specific test
python -m pytest test_model_editing_integration.py::TestREPAIRIntegration::test_repair_initialization
```

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [MODEL_EDITING_README.md](MODEL_EDITING_README.md) | Complete guide |
| [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) | Quick start |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [INDEX.md](INDEX.md) | Component index |

## 🎯 Common Tasks

### Enable REPAIR

```python
agent = MultilingualResearchAgent(enable_repair=True)
```

### Check Statistics

```python
stats = agent.get_repair_statistics()
print(f"Reliability: {stats['editor_stats']['avg_reliability']:.3f}")
```

### Clear History

```python
agent.editor.clear_edit_history()
```

### Health Check

```python
is_healthy = agent.health_checker.is_healthy(query, response)
```

## ⚙️ Environment Variables

```bash
ENABLE_REPAIR=true    # Enable REPAIR integration
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| REPAIR not available | Check `enable_repair=True` |
| Low reliability | Increase `err_thresh` |
| Poor locality | Decrease `mask_ratio` |
| Slow editing | Decrease `num_epochs` |

## 📈 Performance

| Metric | Value |
|--------|-------|
| Edit time | ~1.5s |
| Health check | <0.1s |
| Memory overhead | Minimal |
| Accuracy improvement | +11% |

## 🔗 Integration Points

- ✅ Multilingual Research Agent
- ✅ Quantum/Classical Pipelines
- ✅ Evaluation Harness
- ✅ Health Monitoring
- ✅ Fallback System

## 📞 Support

- **Documentation**: [MODEL_EDITING_README.md](MODEL_EDITING_README.md)
- **Tests**: [test_model_editing_integration.py](test_model_editing_integration.py)
- **Demo**: [demo_repair_integration.py](demo_repair_integration.py)

---

**Version**: 1.1.0 | **Status**: ✅ Complete
