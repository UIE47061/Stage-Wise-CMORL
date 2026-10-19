#!/usr/bin/env bash
# Source from Bash. Override ISAACGYM_VENV for a different installation.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Use: source scripts/activate_env.sh' >&2
    exit 1
fi

_stage_wise_activate() {
    local repo_root env_dir
    repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
    env_dir="${ISAACGYM_VENV:-${VIRTUAL_ENV:-${CONDA_PREFIX:-$repo_root/../deps/isaacgym-preview4/.venv}}}"
    if [[ ! -x "$env_dir/bin/python" ]]; then
        echo 'Set ISAACGYM_VENV to your Isaac Gym Python environment, then source this script again.' >&2
        return 1
    fi
    if [[ -f "$env_dir/bin/activate" ]]; then
        source "$env_dir/bin/activate" || return 1
    else
        export PATH="$env_dir/bin:$PATH"
    fi
    export LD_LIBRARY_PATH="$env_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

if _stage_wise_activate; then
    unset -f _stage_wise_activate
else
    unset -f _stage_wise_activate
    return 1
fi
