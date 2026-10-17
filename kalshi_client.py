Please make the Kalshi service safe for the next code review, then prepare the pending fixes for ChatGPT to review.

Context:

* Repo: Trillionairetone/Trades
* Current commit: 6afa95f2d8dfc008d3abd9373c711df3f1124115
* Kalshi Railway service: victorious-energy
* Start command: python kalshi_bot.py
* Last run was MODE=ordertest on demo. It finished and stopped.
* Another deployment while MODE=ordertest could buy another demo contract.

1. Change the Kalshi service to probe mode

I authorize changing ONLY victorious-energy’s MODE from ordertest to probe. First verify that the deployed probe code cannot place orders, cancel orders, or transfer funds.

Explain whether saving this variable automatically triggers a deployment. If it does, the resulting read-only probe run is authorized. Do not trigger an additional manual restart or deployment.

Verify the saved MODE and, if a probe runs, report its outcome. Keep KALSHI_ENV=demo. Do not expose credentials or modify any other service or setting.

If probe has any trading or funding side effects, stop before changing MODE and explain them.

2. Prepare the pending fixes for review—do not deploy them

Read the actual current code and provide:

* The proposed kalshi_client.py.
* An exact diff against the deployed version.
* SHA256 hashes.
* A concise explanation of every change.
* Offline test results using mocked responses, with no exchange writes.

Review both issues:

* Shard balance units: balances appeared 100× too small, affecting funding amounts as well as display.
* Fees: the demo fill charged $0.0168 for one contract at $0.40.

Verify each API field’s units and applicable fee precision using official documentation and the observed response structure. Do not assume all balance fields share units or generalize the fee rule from one fill. Never include secrets in examples.

3. Review the backtest’s delayed-fill methodology

Reported results:

* Baseline holdout: 1,087 trades, +$29.09, t=2.11.
* Delayed fills: 3,896 trades, −$54.35, t=−2.21.
* Overall verdict: FAIL.

Explain why delaying fills changed the trade count so much. Check whether future quotes are being used to select trades, sides, or checkpoints rather than merely price previously selected orders.

Propose a causal execution-delay test:

* Freeze the original signal time, selected market, side, and decision using information available then.
* Execute that decision one minute later using the appropriate available quote.
* Specify missing-quote, unfillable-order, and price-limit handling in advance.
* Report matched trades and skipped/unfilled decisions separately.

Do not run another historical test, tune rules, or reuse the already-inspected holdout as fresh validation. Explain what untouched data or prospective paper testing would be needed.

Safeguards:

* Only the narrowly authorized MODE change and any automatic read-only probe deployment are allowed.
* No commits, pushes, file deployment, orders, cancellations, transfers, or funding actions.
* Do not change Trades, H4, OANDA, or their settings.
* Do not claim that the successful demo order test proves profitability.

Return:

1. Confirmed service mode and process status.
2. Proposed files, diff, hashes, and offline test evidence.
3. Backtest methodology findings.
4. Any remaining uncertainty or action requiring authorization.

Prepare the concrete review package before asking me to approve deployment.
