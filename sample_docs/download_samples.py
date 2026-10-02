"""Download the sample research papers used for the demo and the evaluation.

The PDFs are not committed to the repository (they belong to their authors);
this script fetches them from arXiv into this folder.

    python sample_docs/download_samples.py
"""
from pathlib import Path
from urllib.request import Request, urlopen

PAPERS = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762v7",
    "bert.pdf": "https://arxiv.org/pdf/1810.04805v2",
    "rag_lewis_2020.pdf": "https://arxiv.org/pdf/2005.11401v4",
}

HERE = Path(__file__).resolve().parent


def main() -> None:
    for name, url in PAPERS.items():
        target = HERE / name
        if target.exists() and target.stat().st_size > 0:
            print(f"already present: {name}")
            continue
        print(f"downloading {name} from {url}")
        req = Request(url, headers={"User-Agent": "PDF-Chats sample downloader"})
        with urlopen(req, timeout=60) as resp:
            target.write_bytes(resp.read())
        print(f"  saved {target.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
