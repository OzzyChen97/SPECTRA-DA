# Safe GitHub publishing checklist

The local repository currently contains public-release commits that may be
ahead of `origin/main`. Publishing should use a credential mechanism that does
not write tokens into tracked files, Git remote URLs, shell history, logs, or
documentation.

## Pre-push audit

Run the lightweight release audit before pushing:

```bash
python scripts/release_audit.py
git status --short --branch
git log --oneline origin/main..HEAD
```

The audit should report `"ok": true`. It checks that `arxiv/main.pdf` is not
older than the LaTeX sources and figure assets. If the audit reports
`arxiv_pdf_freshness`, rebuild the paper with `cd arxiv && ./tools/tectonic
--keep-logs main.tex` before pushing.

## Safe credential options

Use one of these approaches:

1. Use the GitHub CLI interactive login:

   ```bash
   gh auth login
   git push origin main
   ```

2. Use an environment-provided `GH_TOKEN` or `GITHUB_TOKEN` secret that is
   injected by the runtime or CI system, then push with the standard remote:

   ```bash
   git push origin main
   ```

3. Use an OS credential helper or GitHub credential manager configured outside
   the repository, then push normally.

Do not use any method that embeds a token into the remote URL or writes it into
the repository checkout. In particular, do not commit tokens, do not put them
in `.env`, and do not run commands of the form:

```text
git remote set-url origin https://TOKEN@github.com/OWNER/REPO.git
```

## After push

Confirm that the remote received the expected commits and that GitHub Actions
ran the public release audit:

```bash
git fetch origin
git status --short --branch
git log --oneline -5
```

On GitHub, the `release-audit` workflow should pass. This CI only runs
`python scripts/release_audit.py`; it does not require private artifacts,
PyTorch, target labels, or a system LaTeX installation because the paper
freshness check uses the committed PDF timestamp and the repo-local Tectonic
binary when needed.

## If push is still blocked

Export both a commit list and a Git bundle for manual handoff:

```bash
git log --oneline origin/main..HEAD > /tmp/spectra_da_unpushed_commits.txt
git bundle create /tmp/spectra_da_public_unpushed.bundle origin/main..HEAD
```

On a workstation with approved GitHub credentials, apply and push the bundle:

```bash
git clone https://github.com/OzzyChen97/SPECTRA-DA.git spectra-da-publish
cd spectra-da-publish
git pull /tmp/spectra_da_public_unpushed.bundle main
python scripts/release_audit.py
git push origin main
```

If the workstation cannot access the same `/tmp` path, transfer
`/tmp/spectra_da_public_unpushed.bundle` and
`/tmp/spectra_da_unpushed_commits.txt` through the approved file-transfer
mechanism first.
