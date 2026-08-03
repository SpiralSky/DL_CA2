Module src.analysis.analyse_images
==================================
Per-class image statistics for datasets served via a PyTorch DataLoader.

Works for any (image, label) DataLoader where images are equal-sized tensors
(e.g. CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST). No dataset-specific code:
only `num_classes` and `class_names` need to change per dataset.

Functions
---------

`accumulate_pixel_statistics(dataloader, num_classes)`
:   Single pass over the DataLoader, accumulating running sums needed for
    per-class mean/std, without holding the full dataset in memory.
    
    Returns a dict of tensors, each indexed by class:
        count:      (num_classes,)          number of images per class
        sum_:       (num_classes, C)        running per-channel sum
        sumsq:      (num_classes, C)        running per-channel sum of squares
        brightness_sum:    (num_classes,)   running sum of per-image mean brightness
        brightness_sumsq:  (num_classes,)   running sum of squared per-image brightness

`build_stats_dataframe(class_names, count, channel_mean, brightness_mean, brightness_std, texture_scores=None)`
:   Assembles per-class statistics into a pandas DataFrame, one row per class.
    Kept separate from display logic so the raw table can also be inspected,
    filtered, or exported (e.g. df.to_csv()) without touching styling code.

`compute_brightness_mean_std(accum)`
:   Convert accumulated brightness sums into per-class mean and std.
    Returns two (num_classes,) tensors.

`compute_channel_mean_std(accum)`
:   Convert accumulated sums into per-class, per-channel mean and std.
    Returns two (num_classes, C) tensors.

`compute_texture_scores(dataloader, num_classes, samples_per_class=200)`
:   Estimate per-class texture/edge density using Laplacian variance on a
    grayscale-averaged version of each image. Subsamples for speed since this
    runs on CPU via scikit-image rather than as a batched tensor op.
    Returns a (num_classes,) tensor of mean edge variance per class.

`display_class_statistics(df)`
:   Renders the stats table as a styled HTML table in Jupyter, or a plain
    aligned text table otherwise (e.g. running as a plain .py script).

`is_notebook_environment()`
:   Detects whether code is running inside a Jupyter kernel, so display
    logic can choose between rich HTML output and a plain-text fallback.

`plot_channel_means(class_names, channel_mean, channel_names=('R', 'G', 'B'), ax=None)`
:   

`plot_class_balance(class_names, count, ax=None)`
:   

`plot_class_eda_dashboard(class_names, count, channel_mean, texture_scores=None)`
:   Docks the individual plots into a single composite figure. Texture scores
    (which vary meaningfully per class) take the tall left spot; channel means
    and class balance (constant across classes, so demoted) are stacked on
    the right. Each plot function is unchanged and just handed an ax to draw
    on instead of creating its own figure.

`plot_texture_scores(class_names, texture_scores, ax=None)`
:   

`run_class_eda(dataloader, class_names, compute_texture=True, samples_per_class=200)`
:   Orchestrator: runs the full per-class EDA pipeline, prints statistics,
    and displays plots docked into a single dashboard figure.

`style_stats_dataframe(df)`
:   Applies conditional formatting: color gradients on the RGB/brightness
    columns so intensity differences are visible at a glance, and an inline
    bar chart on the texture column so relative magnitude reads instantly.