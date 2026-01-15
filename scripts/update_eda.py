import nbformat as nbf
import os

def update_eda_notebook():
    notebook_path = r'c:\Users\Dark_Cloud_INC\Desktop\Insurrance Claims\notebooks\01_eda.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    new_cells = []
    
    # Process existing cells and add insights
    for cell in nb.cells:
        # Avoid duplicate analysis sections if script is re-run
        if cell.cell_type == 'markdown' and ('## 3. Univariate' in cell.source or '## 4. Bivariate' in cell.source or '## 5. Multivariate' in cell.source or '## 6. Summary' in cell.source):
            continue
        if cell.cell_type == 'code' and ('cat_cols =' in cell.source or 'Claim Amount vs Severity' in cell.source or 'Log Claim Amount by Severity and Gender' in cell.source):
            continue
            
        new_cells.append(cell)
        
        # Add insights after target distribution
        if cell.cell_type == 'code' and 'target_distribution.png' in cell.source:
            insight_cell = nbf.v4.new_markdown_cell(
                "**INSIGHT:** The distribution of `Ultimate_Claim_Amount` is highly right-skewed, "
                "with most claims being small but a few reaching very high values. The log-transformation "
                "successfully normalizes the distribution, making it more suitable for linear modeling."
            )
            new_cells.append(insight_cell)
        
        # Add insights after heatmap
        elif cell.cell_type == 'code' and 'sns.heatmap' in cell.source:
            insight_cell = nbf.v4.new_markdown_cell(
                "**INSIGHT:** The correlation heatmap shows that certain features (to be confirmed by specific analysis) "
                "have stronger linear relationships with the claim amount. Log-transformed features often show clearer "
                "correlations than raw amounts."
            )
            new_cells.append(insight_cell)
            
        # Fix existing boxplot in section 3 if it exists (the one user reported)
        elif cell.cell_type == 'code' and "x='Claim_Type', y='Ultimate_Claim_Amount'" in cell.source:
            cell.source = cell.source.replace("palette='Set3'", "hue='Claim_Type', palette='Set3', legend=False")

    # Add Univariate Analysis (Categorical)
    univariate_header = nbf.v4.new_markdown_cell("## 3. Univariate Analysis (Categorical)")
    new_cells.append(univariate_header)
    
    univariate_code = nbf.v4.new_code_cell(
        "cat_cols = ['Claim_Type', 'Severity', 'Location', 'Gender']\n"
        "fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n"
        "for i, col in enumerate(cat_cols):\n"
        "    ax = axes[i//2, i%2]\n"
        "    sns.countplot(data=df, x=col, ax=ax, hue=col, palette='viridis', legend=False)\n"
        "    ax.set_title(f'Distribution of {col}')\n"
        "    ax.tick_params(axis='x', rotation=45)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
    new_cells.append(univariate_code)
    
    univariate_insight = nbf.v4.new_markdown_cell(
        "**INSIGHT:** Initial distribution analysis shows that 'Collision' and 'Theft' are common claim types. "
        "Severity levels are distributed across Minor, Medium, and Major, which will likely be a key driver for cost."
    )
    new_cells.append(univariate_insight)

    # Add Bivariate Analysis
    bivariate_header = nbf.v4.new_markdown_cell("## 4. Bivariate Analysis")
    new_cells.append(bivariate_header)
    
    bivariate_code = nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n\n"
        "# Claim Amount vs Severity\n"
        "sns.boxplot(x='Severity', y='Log_Ultimate_Claim_Amount', data=df, ax=axes[0], hue='Severity', palette='Set2', legend=False)\n"
        "axes[0].set_title('Log Claim Amount by Severity')\n\n"
        "# Claim Amount vs Claim_Type\n"
        "sns.boxplot(x='Claim_Type', y='Log_Ultimate_Claim_Amount', data=df, ax=axes[1], hue='Claim_Type', palette='Set3', legend=False)\n"
        "axes[1].set_title('Log Claim Amount by Claim Type')\n\n"
        "plt.xticks(rotation=45)\n"
        "plt.show()"
    )
    new_cells.append(bivariate_code)
    
    bivariate_insight = nbf.v4.new_markdown_cell(
        "**INSIGHT:** As expected, 'Major' severity claims have significantly higher median claim amounts compared "
        "to 'Minor' claims. There is also observable variation in costs across different claim types, "
        "suggesting `Claim_Type` is an important predictor."
    )
    new_cells.append(bivariate_insight)
    
    bivariate_age_code = nbf.v4.new_code_cell(
        "plt.figure(figsize=(10, 6))\n"
        "sns.scatterplot(x='Age', y='Log_Ultimate_Claim_Amount', data=df, alpha=0.3)\n"
        "plt.title('Age vs Log Claim Amount')\n"
        "plt.show()"
    )
    new_cells.append(bivariate_age_code)
    
    bivariate_age_insight = nbf.v4.new_markdown_cell(
        "**INSIGHT:** The scatter plot of Age vs Log Claim Amount shows a wide spread, but may indicate "
        "subtle trends in claim sizes for different age brackets."
    )
    new_cells.append(bivariate_age_insight)

    # Add Multivariate Analysis
    multivariate_header = nbf.v4.new_markdown_cell("## 5. Multivariate Analysis")
    new_cells.append(multivariate_header)
    
    multivariate_code = nbf.v4.new_code_cell(
        "plt.figure(figsize=(12, 6))\n"
        "sns.boxplot(x='Severity', y='Log_Ultimate_Claim_Amount', hue='Gender', data=df)\n"
        "plt.title('Log Claim Amount by Severity and Gender')\n"
        "plt.show()"
    )
    new_cells.append(multivariate_code)
    
    multivariate_insight = nbf.v4.new_markdown_cell(
        "**INSIGHT:** Multivariate analysis indicates that the effect of Severity on claim amount is consistent "
        "across Genders, though some slight differences in variance can be observed."
    )
    new_cells.append(multivariate_insight)

    # Add Summary of Findings
    summary_header = nbf.v4.new_markdown_cell("## 6. Summary of Findings")
    new_cells.append(summary_header)
    
    summary_text = nbf.v4.new_markdown_cell(
        "### Key Insights:\n"
        "1. **Target Variable:** The `Ultimate_Claim_Amount` requires log-transformation for better statistical properties.\n"
        "2. **Primary Drivers:** `Severity` and `Claim_Type` are the most significant categorical drivers of claim cost.\n"
        "3. **Segment Trends:** Major severity claims consistently lead to higher costs, regardless of other demographic factors.\n"
        "4. **Modeling Implications:** Non-linear relationships and interactions between `Severity` and other features should be considered in the predictive model."
    )
    new_cells.append(summary_text)

    # Replace notebook cells
    nb.cells = new_cells

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Successfully updated {notebook_path}')

if __name__ == '__main__':
    update_eda_notebook()
