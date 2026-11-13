"""Fixed lower-air coverage follow-up, retaining the original full-range probe."""
from probe_history import main

if __name__ == '__main__':
    main(action_bounds=(-1., 0.))
