#!/usr/bin/env python3
import csv
import re
import os

def update_readme():
    csv_file = "performance_log.csv"
    readme_file = "README.md"

    if not os.path.exists(csv_file):
        print(f"{csv_file} not found. Skipping README update.")
        return

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No data in performance_log.csv")
        return
    
    # Get the latest entry for each model
    latest_entries = {}
    for row in rows:
        latest_entries[row["Model_Name"]] = row
        
    markdown_tables = "Latest benchmark results comparing ColdInfer against standard runtimes. Lower is better for all metrics.\n\n"
    
    for model_name, row in latest_entries.items():
        markdown_tables += f"### {model_name}\n"
        
        metrics = []
        metrics.append({
            "Engine": "ColdInfer",
            "Latency": int(row["Custom_Infer_us"]),
            "Memory": int(row["Custom_RSS_KB"]),
            "CPU": int(row["Custom_CPU_User_us"]) + int(row["Custom_CPU_Sys_us"])
        })
        
        metrics.append({
            "Engine": "TFLM (Micro)",
            "Latency": int(row["TFLM_Infer_us"]) if row.get("TFLM_Infer_us") else 0,
            "Memory": int(row["TFLM_RSS_KB"]) if row.get("TFLM_RSS_KB") else 0,
            "CPU": (int(row["TFLM_CPU_User_us"]) if row.get("TFLM_CPU_User_us") else 0) + (int(row["TFLM_CPU_Sys_us"]) if row.get("TFLM_CPU_Sys_us") else 0)
        })

        metrics.append({
            "Engine": "TFLite",
            "Latency": int(row["TFLite_Infer_us"]),
            "Memory": int(row["TFLite_RSS_KB"]),
            "CPU": int(row["TFLite_CPU_User_us"]) + int(row["TFLite_CPU_Sys_us"])
        })
        
        metrics.append({
            "Engine": "PyTorch",
            "Latency": int(row["PT_Infer_us"]),
            "Memory": int(row["PT_RSS_KB"]),
            "CPU": int(row["PT_CPU_User_us"]) + int(row["PT_CPU_Sys_us"])
        })
        
        # Sort by Memory (RSS), then Latency
        metrics.sort(key=lambda x: (x["Memory"], x["Latency"]))
        
        markdown_tables += "| Engine | Memory RSS (KB) | Latency (µs) | CPU Time (µs) |\n"
        markdown_tables += "|---|---|---|---|\n"
        for m in metrics:
            markdown_tables += f"| **{m['Engine']}** | {m['Memory']} | {m['Latency']} | {m['CPU']} |\n"
        markdown_tables += "\n"

    # Read README.md
    with open(readme_file, "r") as f:
        content = f.read()

    # Regex to find the section between ## Performance Benchmarks and ## Project Roadmap
    pattern = re.compile(r"(## Performance Benchmarks\n\n)(.*?)(?=\n## Project Roadmap & Timeline)", re.DOTALL)
    
    # We replace group 2 with the generated markdown_tables
    # Since sub uses backreferences, we should use a function to avoid escaping issues
    def replacer(match):
        return match.group(1) + markdown_tables

    new_content = pattern.sub(replacer, content)

    with open(readme_file, "w") as f:
        f.write(new_content)
        
    print("README.md performance tables updated successfully.")

if __name__ == "__main__":
    update_readme()
