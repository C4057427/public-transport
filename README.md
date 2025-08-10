# Public Transit Reliability (Pandas + Plotly)

1. Put all real-time CSV into 'data/realtime/', put 'TimetableData.zip' into 'data/' (or extract the XML to 'data/timetable_extracted/').
2. Edit 'config.yaml' (thresholds, column name mappings, etc.) as needed.
3. Run: 'python run_all.py'
4. The result output to `output/`：
   - `metrics_summary.csv`：daily×route×direction OTP、delay statistic
   - `stop_ranking.csv`：Site-level ranking (median delay /OTP)
   - `dashboard.html`：An offline interactive visual dashboard
