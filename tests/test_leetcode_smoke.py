from pathlib import Path


def test_leetcode_repo_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    leetcode_dir = repo_root / "leetcode"
    assert leetcode_dir.exists() and leetcode_dir.is_dir()
    c_files = list(leetcode_dir.glob("*.c"))
    assert len(c_files) >= 1
