# Database integration

`nqo-pg.patch` contains only the PostgreSQL kernel changes required to execute
NQO actions. It adds the `nqo` GUCs, the multi-round Dec/Sched/Enum/Adapt
protocol, query decomposition, join-order enumeration, Filter support, and the
AJoin executor. Python runtime code, experiment scripts, and tests are
excluded.

This is a raw diff from NeurDB commit
`421b2fab467afcab13845a845768cf73c416f5f8` (immediately before the first NQO
integration) to the current kernel source on `neurqo-dev`, including worktree
changes. Names and paths are taken directly from source, without anonymization.
It targets a **NeurDB checkout with a `dbengine/` directory**, not a vanilla
PostgreSQL source checkout. It does not duplicate unrelated NeurDB changes.

Apply it from the root of a NeurDB checkout at that base commit:

```bash
git apply --check /path/to/neurqo/db_integration/nqo-pg.patch
git apply /path/to/neurqo/db_integration/nqo-pg.patch
```

The Filter action additionally requires the Bloom-filter extension in
`db_integration/lip_bloom/`; see its README for build and installation commands.
The integrated NeurDB copy lives in `dbengine/nr_kernel/pg_lip_bloom/` and is
included in the `nr_kernel` build/install targets. `pg_hint_plan` must also be
installed and preloaded.

The AI server, policy models, and experience/training code now also live in
NeurDB under `aiengine/ai_for_db/query_opt/`. They and the extension are separate
from this kernel-only patch; applying the patch does not install them.

Regenerate the same scope from a NeurDB working tree, without name rewriting:

```bash
bash db_integration/export_kernel_patch.sh /path/to/neurdb > db_integration/nqo-pg.patch
```

Already integrated trees should use `git apply --reverse --check` to verify
that the patch is present, rather than applying it twice.
