import pandas as pd

df = pd.read_csv('dataset/financial_events.csv')
reqs = pd.read_csv('dataset/sample_requests.csv')

# How does recurrence inference work in sample answers?
# Check user_03: request_date=2019-09-03, earliest_date=2019-11-15
# No future events - how is the 2019-11-15 income date determined?
u3 = df[df['user_id'] == 'user_03'].sort_values('event_date')
print("=== user_03 all events ===")
cols = ['event_id', 'event_type', 'description', 'category', 'amount', 'event_date', 'settlement_date', 'status', 'flexibility']
print(u3[cols].tail(20).to_string())

print()
# What pattern do income events show?
u3_income = u3[u3['event_type'] == 'income']
print("=== user_03 income events ===")
print(u3_income[['event_id', 'description', 'amount', 'event_date', 'status']].to_string())

print()
# Check user_02: earliest_date=2025-09-15, request_date=2025-08-05
u2 = df[df['user_id'] == 'user_02'].sort_values('event_date')
u2_income = u2[u2['event_type'] == 'income']
print("=== user_02 income events ===")
print(u2_income[['event_id', 'description', 'amount', 'event_date', 'status']].to_string())

print()
# Check pending/scheduled events across all users
pending = df[df['status'].isin(['pending', 'scheduled'])]
print("=== All pending/scheduled events ===")
print(pending[['event_id', 'user_id', 'event_type', 'description', 'amount', 'event_date', 'settlement_date', 'status', 'flexibility']].to_string())
