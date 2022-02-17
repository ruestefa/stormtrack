#!/bin/bash

echo "error: script not tested yet" >&2
exit 1

PREFIX=


bump-patch()
{
    local action="${1}"  # patch, minor, major
    local MSG="${1}"
    echo -e "\n[make bump-${action}] bumping version number: increment ${action} component\n"
    echo -e "\nTag annotation:\n\n${MSG}\n"
    ${PREFIX}bumpversion ${action} --verbose --no-commit --no-tag && echo
    ${PREFIX}pre-commit run --files $(git diff --name-only) && git add -u
    git commit -m "new version v$(cat VERSION) (${action} bump)\n\n$(MSG)" --no-verify && echo
    git tag -a v$$(cat VERSION) -m "$(MSG)"
    echo -e "\ngit tag -n -l v$$(cat VERSION)" && git tag -n -l v$$(cat VERSION)
    echo -e "\ngit log -n1" && git log -n1
}


# TODO
