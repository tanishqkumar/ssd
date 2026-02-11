#!/bin/bash
# Collect results from test suite and produce summary markdown.
# Usage: bash tests/collect_results.sh /path/to/suite_dir
set -e
OUTDIR="${1:?Usage: bash tests/collect_results.sh <suite_dir>}"
SUMMARY="${OUTDIR}/RESULTS.md"

source /home/tkumar/miniconda3/etc/profile.d/conda.sh
conda activate async-spec

cat > "$SUMMARY" << 'HEADER'
# SSD Test Suite Results

Generated automatically by `tests/launch_tests.sh` + `tests/collect_results.sh`.

HEADER

echo "Date: $(date)" >> "$SUMMARY"
echo "Commit: $(cd /home/tkumar/ssd && git rev-parse --short HEAD)" >> "$SUMMARY"
echo "" >> "$SUMMARY"

for logfile in "${OUTDIR}"/*.log; do
    [ -f "$logfile" ] || continue
    BASENAME=$(basename "$logfile")

    echo "---" >> "$SUMMARY"
    echo "" >> "$SUMMARY"
    echo "## $(echo $BASENAME | sed 's/_[0-9]*\.log//' | sed 's/_/ /g')" >> "$SUMMARY"
    echo "Log: \`$BASENAME\`" >> "$SUMMARY"
    echo "" >> "$SUMMARY"

    python3 -c "
import sys, re

with open('$logfile') as f:
    content = f.read()

sections = re.split(r'(=== .+? ===)', content)

i = 1
while i < len(sections):
    header = sections[i].strip().strip('= ')
    body = sections[i+1] if i+1 < len(sections) else ''
    i += 2

    if 'WARMUP' in header or 'ALL DONE' in header:
        continue

    # Extract metrics
    decode = ''
    total = ''
    accepted = ''
    draft_step = ''
    for line in body.split('\n'):
        if 'Decode Throughput' in line and not decode:
            decode = line.strip()
        if 'Total Throughput' in line and not total:
            total = line.strip()
        if 'Avg accepted' in line and not accepted:
            accepted = line.strip()
        if 'draft step' in line and not draft_step:
            draft_step = line.strip()

    # Extract first 2 generations
    gens = []
    gen_blocks = re.findall(r'Prompt (\d+):.*?Generation: (.*?)(?:\n-{20,}|\nPrompt|\Z)', body, re.DOTALL)
    for num, g in gen_blocks[:2]:
        text = g.strip().strip(\"'\").strip('\"')
        if len(text) > 200:
            text = text[:200] + '...'
        gens.append((num, text))

    print(f'### {header}')
    print()
    if decode:
        print(f'**Decode:** \`{decode}\`')
    if total:
        print(f'**Total:** \`{total}\`')
    if accepted:
        print(f'**Accepted:** \`{accepted}\`')
    if draft_step:
        print(f'**Draft step:** \`{draft_step}\`')
    if not decode and not total:
        print(f'**Speed:** (no throughput found — may have crashed)')
    print()
    if gens:
        print('**Sample outputs:**')
        for num, g in gens:
            print(f'{num}. \`{g}\`')
        print()
    else:
        print('**Sample outputs:** (none captured)')
        print()
" >> "$SUMMARY"

done

echo "" >> "$SUMMARY"
echo "---" >> "$SUMMARY"
echo "*End of results.*" >> "$SUMMARY"

echo ""
echo "Summary written to: $SUMMARY"
echo "Preview:"
head -80 "$SUMMARY"
