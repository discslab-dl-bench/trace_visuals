

for d in $(ls data_processed)
do
    mv data_processed/$d/plots_paper/all_combined.png plots/all_paper_plots_v1/${d}.png
    mv data_processed/$d/all_combined/all_combined_overview.png plots/all_paper_plots_v1/${d}_all.png
done