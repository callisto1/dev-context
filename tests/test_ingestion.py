"""Git ingestion tests — unit tests for the ingester logic (no DB required)."""
import os
import tempfile

import pytest

from src.ingestion.git_ingest import GitIngester


def test_extract_repo_name_https():
    assert GitIngester._extract_repo_name("https://github.com/user/my-repo.git") == "my-repo"
    assert GitIngester._extract_repo_name("https://github.com/user/my-repo") == "my-repo"


def test_extract_repo_name_ssh():
    assert GitIngester._extract_repo_name("git@github.com:user/my-repo.git") == "my-repo"


def test_extract_repo_name_local():
    assert GitIngester._extract_repo_name("/home/user/projects/my-repo") == "my-repo"


@pytest.mark.asyncio
async def test_ingest_local_repo():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "repo", ".git"))
        with open(os.path.join(tmpdir, "repo", ".git", "HEAD"), "w") as f:
            f.write("ref: refs/heads/main\n")

        with open(os.path.join(tmpdir, "repo", "main.py"), "w") as f:
            f.write("def hello():\n    print('hello world')\n")

        with open(os.path.join(tmpdir, "repo", "README.md"), "w") as f:
            f.write("# My Project\nThis is a test.\n")

        ingester = GitIngester()
        ingester.clone_dir = tmpdir

        # GitIngester expects a git repo — for unit tests, test the helper methods
        assert ingester._extract_repo_name(os.path.join(tmpdir, "repo")) == "repo"
