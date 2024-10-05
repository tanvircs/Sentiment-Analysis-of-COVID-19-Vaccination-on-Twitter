# Sentiment-Analysis-of-COVID-19-Vaccination-on-Twitter
# Power BI Sales Dashboard Project

## Project Objective
The objective of this Power BI report is to analyze and present comprehensive insights into sales, profit, orders, profit margin, and various comparisons to provide a clear understanding of key performance indicators and trends.

### Report Objectives
- **Calculate Total Sales**: Display the total sales value for the selected period.
- **Calculate Profit**: Visualize the total profit achieved.
- **Analyze Orders**: Examine the number of orders placed.
- **Calculate Profit Margin**: Visualize the profit margin percentage.
- **Compare Sales by Product**: With Previous Year.
- **Compare Sales by Months**: With Previous Year.
- **Display Top 5 Cities**: Based on sales.
- **Compare Profit by Channel**: With Previous Year.
- **Analyze Sales by Customer**: And compare with the previous year.
- **Create Slicers**: For Date, City, Product, and Channel.

## Steps for Power BI Project
1. **Gather Data**: Collect necessary data from various sources.
2. **Power Query – Data ETL**: Clean and transform the data using Power Query Editor.
3. **Create a Date Table**: Essential for DAX time intelligence functions.
4. **Create Data Model**: In Power BI Desktop, representing relationships between tables.
5. **Develop Reports**: Use Power BI Desktop to create reports with visualizations.
    - Create report background in PowerPoint.
    - Implement slicers for Date, City, Product, and Channel.
    - Develop DAX measures.
    - Create visuals for sales, profit, and more.

### DAX Calculations Examples
```dax
Sales = SUM(Sales_Data[Sales])
Sales PY = CALCULATE([Sales], SAMEPERIODLASTYEAR(DateTable[Date]))
Sales vs PY = [Sales] - [Sales PY]
Sales vs py % = DIVIDE([Sales vs PY],[Sales],0)
Products Sold = SUM(Sales_Data[Order Quantity])
Profit = SUM(Sales_Data[Profit])
...
