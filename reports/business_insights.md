# Business insights and recommendations

Every number below traces to a file in `reports/` or `images/`. Forecast window: 2018-01-01 to 2018-01-28.

| Area | Finding | Source | Recommended action |
|---|---|---|---|
| Overall demand | 605,363 units forecast (80% range about 494,000 to 723,000), +8.2% vs the same weeks of 2017, -2.5% vs the last 28 days | `insights_summary.json` | Set the January target from the forecast range. Treat +8.2% as an upper estimate: validation over-forecast Q1 2017 by about 2%, and 2017 grew 3.9%. |
| Staffing | Sunday index 119, Saturday 112, Monday 79. Peak forecast days are Sundays 7, 21 and 28 January | `eda_findings.json` | Roster to the weekday profile and pre-plan cover for Sundays and Saturdays. |
| Seasonal peaks | Weeks 26 to 30 have index 123 to 129. Weeks 1 to 4 have about 68 | `weekly_seasonal_index.csv` | Place purchase orders ahead of week 26 allowing for lead time. Defer orders in January. |
| Inventory | Safety stock is about 9% of 7-day lead-time demand at 95% service | `inventory_policy_example.csv` | Reorder point = lead-time demand + safety stock. Use the dashboard reorder planner with real stock and supplier lead time. |
| Uncertain items | Items 5, 41, 27, 47, 1 have ranges 61% to 66% of forecast | `item_insights.csv` | Hold a larger safety stock and review these weekly. |
| Growth | No item is forecast to decline. Items 2 (+12.0%), 40, 38, 18, 30 grow fastest. Items 5, 34, 41, 49, 1 grow slowest (+4% to +6%) | `item_insights.csv` | Prioritise supply for the growth items. There are no de-stocking candidates in this window. |
| Slow movers | Items 5, 1, 41, 4, 47 have the lowest forecast volume (4,263 to 5,081 units over 28 days) | `item_insights.csv` | Check holding cost against margin. No margin data was available, so no de-stocking call is made. |
| Marketing | Demand is lowest in January and February | `eda_findings.json` | Concentrate promotions there and avoid discounting in summer peak weeks. |
