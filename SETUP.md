# Environment setup and data download

Everything you need to run the RETECO sub-track 2a code on your machine. It takes about 15 minutes, mostly waiting for installs.

**You need:** Git, access to this repo, and a few GB of free disk space. Steps 1–6 are for **macOS** with [Homebrew](https://brew.sh). **On Windows, skip to [Windows setup](#windows-setup)**; Linux notes are at the end.

## 1. Clone the repo

```bash
git clone https://github.com/glenpaulson/nlp-semeval.git
cd nlp-semeval
```

Or in GitHub Desktop: **File → Clone repository**.

## 2. Install uv

```bash
brew install uv
```

## 3. Create the Python environment

```bash
uv sync
```

This creates `.venv/` with Python 3.12 and the exact package versions in `uv.lock`. uv downloads Python 3.12 itself if you don't have it. The first run takes a few minutes because Pyserini pulls in PyTorch.

## 4. Install Java 21

Pyserini runs the Lucene analyzer in Java. We need it to reproduce the official BM25 baseline exactly.

```bash
brew install openjdk@21
echo 'export JAVA_HOME="$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home"' >> ~/.zshrc
echo 'export JVM_PATH="$JAVA_HOME/lib/server/libjvm.dylib"' >> ~/.zshrc
source ~/.zshrc
```

## 5. Check the setup

```bash
uv run python -c "from pyserini.analysis import Analyzer, get_lucene_analyzer; print(Analyzer(get_lucene_analyzer()).analyze('Running drones'))"
```

You should see `['run', 'drone']`. That means Python, Pyserini and Java are connected.

## 6. Download the data (Track 2 only)

```bash
uv run hf download DataScience-UIBK/RETECO-SemEval2027 --repo-type dataset \
  --include "track2_recor/*" --include "split_manifest.json" \
  --local-dir reteco_data
```

- Repeat `--include` for each pattern. With one `--include` followed by two patterns, the second one is read as a file name.
- No login needed: the dataset is public. Official page: [DataScience-UIBK/RETECO-SemEval2027](https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027).

You should end up with:

```text
reteco_data/
  split_manifest.json
  track2_recor/<domain>/
    documents.jsonl        corpus to search: {doc_id, content}
    benchmark_train.json   conversations: turns, query, conversation_history, gold_doc_ids
    benchmark_dev.json
    qrels_train.txt        relevance labels; topic id = <conv_id>_turn_<turn_id>
    qrels_dev.txt
```

There are 11 domains: biology, drones, earth_science, economics, hardware, law, medicalsciences, politics, psychology, robotics, sustainable_living. Across them: 496 train / 211 dev conversations and 2,113 train / 858 dev turns.

**Never commit `reteco_data/`.** It is in `.gitignore` because the files are too big for GitHub (100 MB limit per file), anything committed stays in git history forever, and everyone can download identical files with the command above.

## Daily use

- **Run a script:** `uv run python path/to/script.py`, or activate the environment once with `source .venv/bin/activate`.
- **After pulling teammates' changes:** run `uv sync` if `uv.lock` changed.
- **Add a package:** `uv add <package>`, then regenerate `requirements.txt`:
  ```bash
  uv export --format requirements-txt --no-hashes -o requirements.txt
  ```
  Commit `pyproject.toml`, `uv.lock` and `requirements.txt` together.
- **API keys:** put them in `.env` in the repo root. It is gitignored; never put keys in code.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `pytrec_eval` fails to build | Run `xcode-select --install`, then `uv sync` again. Still failing: `uv add pytrec-eval-terrier` (same `import pytrec_eval`). |
| Pyserini can't find the JVM | `echo $JAVA_HOME $JVM_PATH` must print real paths. Open a new terminal after editing `~/.zshrc`. |
| `unexpected argument '#'` when pasting commands | zsh doesn't treat `#` as a comment by default: `echo 'setopt interactive_comments' >> ~/.zshrc` |
| `hf: command not found` | Use `uv run hf ...`; `hf` is installed inside `.venv` with `huggingface_hub`. |

## Linux and Windows

- **Linux:** install uv with `curl -LsSf https://astral.sh/uv/install.sh | sh` and Java with `sudo apt install openjdk-21-jdk`. Set `JVM_PATH=$JAVA_HOME/lib/server/libjvm.so`.
- **Windows:** the easiest route is WSL2 (Ubuntu), then follow the Linux steps.
