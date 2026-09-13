import sys
import pandas as pd
from datetime import datetime, date
from cashflow import simulate, ScheduledPayment, SpendingCut
from financial_state import build_financial_state
from data_loader import DataStore

def main():
    print("Loading data using DataStore...")
    store = DataStore()
    df = pd.read_csv("output.csv")
    print("Data loaded. Starting mathematical validation...")
    
    boundary_pass = 0
    boundary_fail = 0
    installment_pass = 0
    installment_fail = 0
    replay_pass = 0
    replay_fail = 0
    
    for idx, row in df.iterrows():
        req_id = row['request_id']
        req_row = store.requests_df[store.requests_df['request_id'] == req_id]
        if len(req_row) == 0:
            print(f"[{req_id}] Not found in requests dataset.")
            continue
        req_row = req_row.iloc[0]
        
        bundle = store.get_bundle(req_row)
        if not bundle.profile:
            continue
            
        # Build canonical state
        state = build_financial_state(bundle.profile, bundle.events, bundle.request_date, {}, None)
        
        # 1. Boundary Test
        amount_safe = row['amount_safe_to_pay']
        requested = bundle.requested_amount
        
        if amount_safe < requested:
            # Test X is SAFE
            res_safe = simulate(state, [ScheduledPayment(bundle.request_date, amount_safe)], [], bundle.request_date)
            # Test X + 0.01 is UNSAFE
            res_unsafe = simulate(state, [ScheduledPayment(bundle.request_date, amount_safe + 0.01)], [], bundle.request_date)
            if res_safe.is_safe and not res_unsafe.is_safe:
                boundary_pass += 1
            else:
                boundary_fail += 1
                if not res_safe.is_safe:
                    print(f"[{req_id}] Boundary FAIL: X ({amount_safe}) is UNSAFE. Margin: {res_safe.safety_margin}")
                if res_unsafe.is_safe:
                    print(f"[{req_id}] Boundary FAIL: X+0.01 ({amount_safe + 0.01}) is SAFE.")

        # 2. Installment Validation
        if row['recommended_payment_method'] == 'installments':
            plan_str = row['payment_plan']
            matched = False
            for opt in bundle.payment_options:
                if opt.payment_method == "installments":
                    sched = opt.generate_schedule()
                    if sched:
                        sched_str = "|".join(f"{d.isoformat()}:{round(a, 2)}" for d, a in sched)
                        if sched_str == plan_str:
                            matched = True
                            break
            if matched:
                installment_pass += 1
            else:
                installment_fail += 1
                print(f"[{req_id}] Installment FAIL: generated {plan_str} does not match any supplied payment option.")

        # 3. Full Strategy Safety Replay
        method = row['recommended_payment_method']
        plan_str = row['payment_plan']
        cuts_str = row['spending_changes_needed']
        
        if method != 'not_recommended':
            # Reconstruct payments
            payments = []
            if plan_str != "none":
                for p in plan_str.split('|'):
                    d_str, a_str = p.split(':')
                    payments.append(ScheduledPayment(date.fromisoformat(d_str), float(a_str)))
            
            # Reconstruct cuts
            cuts = []
            if cuts_str != "none":
                for c in cuts_str.split('|'):
                    parts = c.split(':')
                    ctype = parts[0]
                    ev_id = parts[1]
                    new_amt = 0.0 if ctype == "stop" else float(parts[2])
                    cuts.append(SpendingCut(ev_id, 0.0, new_amt, ctype, "placeholder", "placeholder"))
            
            # Replay
            res_replay = simulate(state, payments, cuts, bundle.request_date)
            if res_replay.is_safe:
                replay_pass += 1
            else:
                replay_fail += 1
                print(f"[{req_id}] Replay FAIL: method {method}, min balance: {res_replay.safety_margin}")

    print("\n=== Validation Results ===")
    print(f"Boundary Test (X vs X+0.01)    : {boundary_pass} passed, {boundary_fail} failed")
    print(f"Installment Strict Match       : {installment_pass} passed, {installment_fail} failed")
    print(f"Full Strategy Replay Safety    : {replay_pass} passed, {replay_fail} failed")

if __name__ == '__main__':
    main()
