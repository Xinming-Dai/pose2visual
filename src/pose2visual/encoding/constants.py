"""Constants shared by the neural-data extraction and encoding steps."""

# keypoint-derived IBL behavior traces (hyphenated names, as produced by step 0); the
# <eid>_aligned.npz keys use underscores instead of hyphens
DYNAMIC_VARS = (
    'wheel-speed',
    'licks',
    'left-whisker-motion-energy',
    'right-whisker-motion-energy',
    'left-nose-speed',
    'right-nose-speed',
    'left-paw-speed',
    'right-paw-speed',
)
