# Label rule: how every complaint category is treated

This page explains what the models predict and how each CFPB complaint category is labelled. The rule itself lives in [`configs/categories.yaml`](../configs/categories.yaml); the `label` stage applies it and refuses to run if the file and the data disagree.

## What is predicted

Each complaint in the six target products gets one of three treatments, based only on the **Issue and Sub-issue the consumer chose** when filing:

- **Fraud (1):** the category describes fraud, a scam, identity theft, or an unauthorized transaction.
- **Not fraud (0):** any other category in the target products.
- **Excluded:** the category is too ambiguous to call either way, so its complaints are removed before sampling. They are never in the training or test set.

The label is the consumer's own category choice, not verified fraud. The Issue and Sub-issue fields are used only to build the label and never reach any model.

## Version history

| Version | Date | Change |
|---|---|---|
| 1 | 2026-09 | Reviewed allow-list. Fraud-sounding categories that are not fraud events were labelled 0 ("reviewed negatives"). |
| **2** | **2026-09-29** | Category-by-category review, decided **on the meaning of each category only, before looking at any model errors**. "Debt is not yours" and "Impersonated attorney, law enforcement, or government official" became fraud. Ambiguous categories were excluded, including all former reviewed negatives. |

Results from the two versions are not comparable: they predict different targets, and the no-skill PR-AUC (the fraud rate) moves from about 0.17 to about 0.29.

## Version 2 review decisions

Every category someone could argue belongs on the other side was reviewed. The case for each side is summarised; the decision is final for v2.

| # | Category | Complaints | v1 | v2 | Reasoning |
|---|---|---:|---|---|---|
| N1 | Debt collection · Attempts to collect debt not owed · **Debt is not yours** | 36,925 | 0 | **Fraud** | Its sibling "Debt was result of identity theft" is fraud; which box a consumer ticks between the two is close to arbitrary. |
| N2 | Debt collection · False statements or representation · **Impersonated attorney, law enforcement, or government official** | 2,135 | 0 | **Fraud** | Impersonation is a hallmark of scam ("phantom debt") collectors. |
| N3 | Credit card (or prepaid card) · Getting a credit card · **Sent card you never applied for** | 1,004 | 0 | **Excluded** | Can be identity theft, or an unsolicited offer or issuer replacement. |
| N4/N5 | Credit/prepaid card · **Company isn't resolving a dispute about a purchase (or transfer)** | 17,836 | 0 | **Excluded** | Sometimes a disputed unauthorized charge; mostly merchant quality, refunds and returns. |
| N6 | Checking · Managing an account · Deposits and withdrawals | 12,588 | 0 | Not fraud | Mostly holds, delays and deposit errors. |
| N7 | Checking · Managing an account · Problem using a debit or ATM card | 7,453 | 0 | Not fraud | Mostly declined cards and ATM errors. |
| N8 | Checking · Managing an account · Problem accessing account | 3,679 | 0 | Not fraud | Mostly lockouts and verification problems. |
| N9 | Checking · Managing an account · Funds not handled or disbursed as instructed | 4,897 | 0 | Not fraud | Mostly processing errors. |
| N10 | Checking · Company charging your account · Can't stop withdrawals from your account | 711 | 0 | Not fraud | Usually subscriptions or loans the consumer agreed to. |
| N11 | Money transfer · Other transaction problem | 4,945 | 0 | Not fraud | Mostly failed, delayed or reversed transfers. |
| N12 | Money transfer · Managing, opening, or closing your mobile wallet account | 3,578 | 0 | Not fraud | Mostly account freezes, closures and verification. |
| R1 | Incorrect information on your report · **Information belongs to someone else** | 596 | 0 | **Excluded** | Can be identity theft, often a mixed-up credit file. |
| R2 | Improper use of your report · **Credit inquiries you don't recognize** | 223 | 0 | **Excluded** | Can be fraudulent applications, often forgotten legitimate inquiries. |
| R3 | Money transfer · **Lost or stolen check / money order** | 331 | 0 | **Excluded** | "Lost" is not fraud but "stolen" is; the category does not separate them. |
| R4 | **Fraud alerts or security freezes**; **credit monitoring or identity theft protection services** | 276 | 0 | **Excluded** | About the protection product itself, not a fraud event. |
| P1 | Money transfer · Unauthorized transactions or other transaction problem | 2,990 | 1 | Fraud | Bundles other problems, but unauthorized transactions are its core. |
| P2 | Checking · Company charging your account · Transaction was not authorized | 6,822 | 1 | Fraud | Literally "not authorized". |
| P3 | Credit/prepaid card · Charged for something you did not purchase / purchase you did not make | 8,118 | 1 | Fraud | Literally unauthorized charges. |
| P4 | Debt collection · Attempts to collect debt not owed · Debt was result of identity theft | 22,562 | 1 | Fraud | Literally names identity theft. |

Not reviewed because there was no real case for switching: Money transfer "Fraud or scam", and the "card or account opened as a result of identity theft / fraud / without consent" categories (all fraud).

## Effect on the data (v2)

Counts are for complaints with a narrative received May 2018 to August 2023 in the target products (CFPB archives 2–4), **before** near-duplicate removal and sampling. The model sample is 30,000 complaints drawn from these at the natural fraud rate.

| Product | Fraud (1) | Not fraud (0) | Excluded | Fraud rate |
|---|---:|---:|---:|---:|
| Debt collection | 61,622 | 94,108 | 0 | 39.6% |
| Money transfer, virtual currency, or money service | 14,316 | 15,942 | 331 | 47.3% |
| Credit card or prepaid card | 12,958 | 64,062 | 19,640 | 16.8% |
| Checking or savings account | 9,755 | 61,303 | 166 | 13.7% |
| Credit card | 64 | 600 | 124 | 9.6% |
| Prepaid card | 8 | 35 | 5 | 18.6% |
| **Total** | **98,723** | **236,050** | **20,266** | **29.5%** |

## Every category in the target products

Sorted by product, then treatment (fraud, excluded, not fraud), then size. Generated from the loaded complaints with the v2 rule; "—" means the category has no sub-issue.

| Product | Issue | Sub-issue | Complaints | Treatment |
|---|---|---|---:|---|
| Checking or savings account | Problem with a lender or other company charging your account | Transaction was not authorized | 6,822 | Fraud (1) |
| Checking or savings account | Opening an account | Account opened as a result of fraud | 2,917 | Fraud (1) |
| Checking or savings account | Opening an account | Account opened without my consent or knowledge | 16 | Fraud (1) |
| Checking or savings account | Incorrect information on your report | Information belongs to someone else | 59 | Excluded |
| Checking or savings account | Problem with fraud alerts or security freezes | — | 57 | Excluded |
| Checking or savings account | Improper use of your report | Credit inquiries on your report that you don't recognize | 13 | Excluded |
| Checking or savings account | Credit monitoring or identity theft protection services | Billing dispute for services | 12 | Excluded |
| Checking or savings account | Credit monitoring or identity theft protection services | Problem canceling credit monitoring or identify theft protection service | 10 | Excluded |
| Checking or savings account | Credit monitoring or identity theft protection services | Didn't receive services that were advertised | 7 | Excluded |
| Checking or savings account | Credit monitoring or identity theft protection services | Problem with product or service terms changing | 5 | Excluded |
| Checking or savings account | Credit monitoring or identity theft protection services | Received unwanted marketing or advertising | 3 | Excluded |
| Checking or savings account | Managing an account | Deposits and withdrawals | 12,588 | Not fraud (0) |
| Checking or savings account | Managing an account | Problem using a debit or ATM card | 7,453 | Not fraud (0) |
| Checking or savings account | Managing an account | Funds not handled or disbursed as instructed | 4,897 | Not fraud (0) |
| Checking or savings account | Managing an account | Banking errors | 4,446 | Not fraud (0) |
| Checking or savings account | Closing an account | Company closed your account | 4,074 | Not fraud (0) |
| Checking or savings account | Managing an account | Problem accessing account | 3,679 | Not fraud (0) |
| Checking or savings account | Closing an account | Funds not received from closed account | 3,436 | Not fraud (0) |
| Checking or savings account | Managing an account | Fee problem | 2,811 | Not fraud (0) |
| Checking or savings account | Problem caused by your funds being low | Overdrafts and overdraft fees | 2,673 | Not fraud (0) |
| Checking or savings account | Managing an account | Problem making or receiving payments | 2,564 | Not fraud (0) |
| Checking or savings account | Opening an account | Didn't receive terms that were advertised | 2,381 | Not fraud (0) |
| Checking or savings account | Closing an account | Can't close your account | 2,360 | Not fraud (0) |
| Checking or savings account | Opening an account | Unable to open an account | 2,065 | Not fraud (0) |
| Checking or savings account | Problem caused by your funds being low | Non-sufficient funds and associated fees | 1,184 | Not fraud (0) |
| Checking or savings account | Problem with a lender or other company charging your account | Money was taken from your account on the wrong day or for the wrong amount | 1,125 | Not fraud (0) |
| Checking or savings account | Managing an account | Cashing a check | 984 | Not fraud (0) |
| Checking or savings account | Problem with a lender or other company charging your account | Can't stop withdrawals from your account | 711 | Not fraud (0) |
| Checking or savings account | Closing an account | Fees charged for closing account | 482 | Not fraud (0) |
| Checking or savings account | Problem caused by your funds being low | Bounced checks or returned payments | 396 | Not fraud (0) |
| Checking or savings account | Opening an account | Confusing or missing disclosures | 348 | Not fraud (0) |
| Checking or savings account | Problem caused by your funds being low | Late or other fees | 243 | Not fraud (0) |
| Checking or savings account | Managing an account | Deposits or withdrawals | 145 | Not fraud (0) |
| Checking or savings account | Managing an account | Problem with fees or penalties | 52 | Not fraud (0) |
| Checking or savings account | Incorrect information on your report | Account status incorrect | 42 | Not fraud (0) |
| Checking or savings account | Managing an account | Problem with renewal | 36 | Not fraud (0) |
| Checking or savings account | Improper use of your report | Reporting company used your report improperly | 23 | Not fraud (0) |
| Checking or savings account | Incorrect information on your report | Account information incorrect | 21 | Not fraud (0) |
| Checking or savings account | Problem with a credit reporting company's investigation into an existing problem | Investigation took more than 30 days | 17 | Not fraud (0) |
| Checking or savings account | Problem with a credit reporting company's investigation into an existing problem | Their investigation did not fix an error on your report | 14 | Not fraud (0) |
| Checking or savings account | Incorrect information on your report | Personal information incorrect | 9 | Not fraud (0) |
| Checking or savings account | Problem with a credit reporting company's investigation into an existing problem | Was not notified of investigation status or results | 7 | Not fraud (0) |
| Checking or savings account | Problem with a credit reporting company's investigation into an existing problem | Problem with personal statement of dispute | 7 | Not fraud (0) |
| Checking or savings account | Incorrect information on your report | Old information reappears or never goes away | 7 | Not fraud (0) |
| Checking or savings account | Problem with a credit reporting company's investigation into an existing problem | Difficulty submitting a dispute or getting information about a dispute over the phone | 6 | Not fraud (0) |
| Checking or savings account | Unable to get your credit report or credit score | Other problem getting your report or credit score | 5 | Not fraud (0) |
| Checking or savings account | Incorrect information on your report | Public record information inaccurate | 3 | Not fraud (0) |
| Checking or savings account | Improper use of your report | Received unsolicited financial product or insurance offers after opting out | 3 | Not fraud (0) |
| Checking or savings account | Incorrect information on your report | Information is missing that should be on the report | 2 | Not fraud (0) |
| Checking or savings account | Unable to get your credit report or credit score | Problem getting your free annual credit report | 2 | Not fraud (0) |
| Checking or savings account | Getting a line of credit | — | 1 | Not fraud (0) |
| Checking or savings account | Improper use of your report | Report provided to employer without your written authorization | 1 | Not fraud (0) |
| Credit card | Getting a credit card | Card opened without my consent or knowledge | 36 | Fraud (1) |
| Credit card | Problem with a purchase shown on your statement | Card was charged for something you did not purchase with the card | 28 | Fraud (1) |
| Credit card | Problem with a purchase shown on your statement | Credit card company isn't resolving a dispute about a purchase on your statement | 104 | Excluded |
| Credit card | Getting a credit card | Sent card you never applied for | 9 | Excluded |
| Credit card | Incorrect information on your report | Information belongs to someone else | 6 | Excluded |
| Credit card | Improper use of your report | Credit inquiries on your report that you don't recognize | 2 | Excluded |
| Credit card | Credit monitoring or identity theft protection services | Problem canceling credit monitoring or identify theft protection service | 1 | Excluded |
| Credit card | Credit monitoring or identity theft protection services | Problem with product or service terms changing | 1 | Excluded |
| Credit card | Credit monitoring or identity theft protection services | Didn't receive services that were advertised | 1 | Excluded |
| Credit card | Problem with a company's investigation into an existing problem | Was not notified of investigation status or results | 255 | Not fraud (0) |
| Credit card | Fees or interest | Problem with fees | 46 | Not fraud (0) |
| Credit card | Other features, terms, or problems | Other problem | 42 | Not fraud (0) |
| Credit card | Problem when making payments | Problem during payment process | 34 | Not fraud (0) |
| Credit card | Closing your account | Company closed your account | 30 | Not fraud (0) |
| Credit card | Trouble using your card | Can't use card to make purchases | 21 | Not fraud (0) |
| Credit card | Fees or interest | Charged too much interest | 18 | Not fraud (0) |
| Credit card | Advertising and marketing, including promotional offers | Didn't receive advertised or promotional terms | 18 | Not fraud (0) |
| Credit card | Advertising and marketing, including promotional offers | Confusing or misleading advertising about the credit card | 14 | Not fraud (0) |
| Credit card | Other features, terms, or problems | Problem with rewards from credit card | 13 | Not fraud (0) |
| Credit card | Getting a credit card | Application denied | 13 | Not fraud (0) |
| Credit card | Trouble using your card | Credit card company won't increase or decrease your credit limit | 11 | Not fraud (0) |
| Credit card | Problem with a purchase shown on your statement | Overcharged for something you did purchase with the card | 11 | Not fraud (0) |
| Credit card | Closing your account | Can't close your account | 9 | Not fraud (0) |
| Credit card | Improper use of your report | Reporting company used your report improperly | 7 | Not fraud (0) |
| Credit card | Struggling to pay your bill | Credit card company won't work with you while you're going through financial hardship | 6 | Not fraud (0) |
| Credit card | Problem with a company's investigation into an existing problem | Their investigation did not fix an error on your report | 6 | Not fraud (0) |
| Credit card | Other features, terms, or problems | Problem with customer service | 6 | Not fraud (0) |
| Credit card | Incorrect information on your report | Account information incorrect | 5 | Not fraud (0) |
| Credit card | Fees or interest | Unexpected increase in interest rate | 4 | Not fraud (0) |
| Credit card | Other features, terms, or problems | Privacy issues | 4 | Not fraud (0) |
| Credit card | Other features, terms, or problems | Credit card company forcing arbitration | 3 | Not fraud (0) |
| Credit card | Problem with a company's investigation into an existing problem | Investigation took more than 30 days | 3 | Not fraud (0) |
| Credit card | Problem when making payments | You never received your bill or did not know a payment was due | 3 | Not fraud (0) |
| Credit card | Getting a credit card | Delay in processing application | 3 | Not fraud (0) |
| Credit card | Other features, terms, or problems | Problem with balance transfer | 3 | Not fraud (0) |
| Credit card | Problem with a company's investigation into an existing problem | Difficulty submitting a dispute or getting information about a dispute over the phone | 3 | Not fraud (0) |
| Credit card | Incorrect information on your report | Account status incorrect | 2 | Not fraud (0) |
| Credit card | Struggling to pay your bill | Filed for bankruptcy | 2 | Not fraud (0) |
| Credit card | Other features, terms, or problems | Problem with cash advances | 2 | Not fraud (0) |
| Credit card | Getting a credit card | Problem getting a working replacement card | 1 | Not fraud (0) |
| Credit card | Struggling to pay your bill | Problem lowering your monthly payments | 1 | Not fraud (0) |
| Credit card | Incorrect information on your report | Personal information incorrect | 1 | Not fraud (0) |
| Credit card or prepaid card | Problem with a purchase shown on your statement | Card was charged for something you did not purchase with the card | 7,011 | Fraud (1) |
| Credit card or prepaid card | Getting a credit card | Card opened as result of identity theft or fraud | 4,868 | Fraud (1) |
| Credit card or prepaid card | Problem with a purchase or transfer | Charged for a purchase or transfer you did not make with the card | 1,079 | Fraud (1) |
| Credit card or prepaid card | Problem with a purchase shown on your statement | Credit card company isn't resolving a dispute about a purchase on your statement | 16,001 | Excluded |
| Credit card or prepaid card | Problem with a purchase or transfer | Card company isn't resolving a dispute about a purchase or transfer | 1,726 | Excluded |
| Credit card or prepaid card | Getting a credit card | Sent card you never applied for | 995 | Excluded |
| Credit card or prepaid card | Incorrect information on your report | Information belongs to someone else | 531 | Excluded |
| Credit card or prepaid card | Improper use of your report | Credit inquiries on your report that you don't recognize | 208 | Excluded |
| Credit card or prepaid card | Credit monitoring or identity theft protection services | Billing dispute for services | 71 | Excluded |
| Credit card or prepaid card | Problem with fraud alerts or security freezes | — | 51 | Excluded |
| Credit card or prepaid card | Credit monitoring or identity theft protection services | Didn't receive services that were advertised | 18 | Excluded |
| Credit card or prepaid card | Credit monitoring or identity theft protection services | Problem with product or service terms changing | 18 | Excluded |
| Credit card or prepaid card | Credit monitoring or identity theft protection services | Problem canceling credit monitoring or identify theft protection service | 13 | Excluded |
| Credit card or prepaid card | Credit monitoring or identity theft protection services | Received unwanted marketing or advertising | 8 | Excluded |
| Credit card or prepaid card | Problem when making payments | Problem during payment process | 6,520 | Not fraud (0) |
| Credit card or prepaid card | Fees or interest | Problem with fees | 6,453 | Not fraud (0) |
| Credit card or prepaid card | Other features, terms, or problems | Other problem | 5,163 | Not fraud (0) |
| Credit card or prepaid card | Closing your account | Company closed your account | 4,890 | Not fraud (0) |
| Credit card or prepaid card | Problem with a credit reporting company's investigation into an existing problem | Was not notified of investigation status or results | 3,999 | Not fraud (0) |
| Credit card or prepaid card | Advertising and marketing, including promotional offers | Didn't receive advertised or promotional terms | 3,061 | Not fraud (0) |
| Credit card or prepaid card | Trouble using your card | Can't use card to make purchases | 2,651 | Not fraud (0) |
| Credit card or prepaid card | Getting a credit card | Application denied | 2,626 | Not fraud (0) |
| Credit card or prepaid card | Other features, terms, or problems | Problem with rewards from credit card | 2,601 | Not fraud (0) |
| Credit card or prepaid card | Fees or interest | Charged too much interest | 2,293 | Not fraud (0) |
| Credit card or prepaid card | Advertising and marketing, including promotional offers | Confusing or misleading advertising about the credit card | 2,069 | Not fraud (0) |
| Credit card or prepaid card | Unexpected or other fees | — | 1,727 | Not fraud (0) |
| Credit card or prepaid card | Closing your account | Can't close your account | 1,652 | Not fraud (0) |
| Credit card or prepaid card | Problem when making payments | You never received your bill or did not know a payment was due | 1,644 | Not fraud (0) |
| Credit card or prepaid card | Trouble using your card | Credit card company won't increase or decrease your credit limit | 1,403 | Not fraud (0) |
| Credit card or prepaid card | Other features, terms, or problems | Problem with customer service | 1,313 | Not fraud (0) |
| Credit card or prepaid card | Struggling to pay your bill | Credit card company won't work with you while you're going through financial hardship | 1,309 | Not fraud (0) |
| Credit card or prepaid card | Problem with a credit reporting company's investigation into an existing problem | Their investigation did not fix an error on your report | 1,249 | Not fraud (0) |
| Credit card or prepaid card | Trouble using the card | Trouble using the card to spend money in a store or online | 1,190 | Not fraud (0) |
| Credit card or prepaid card | Fees or interest | Unexpected increase in interest rate | 771 | Not fraud (0) |
| Credit card or prepaid card | Incorrect information on your report | Account status incorrect | 744 | Not fraud (0) |
| Credit card or prepaid card | Incorrect information on your report | Account information incorrect | 732 | Not fraud (0) |
| Credit card or prepaid card | Other features, terms, or problems | Problem with balance transfer | 727 | Not fraud (0) |
| Credit card or prepaid card | Problem with a purchase shown on your statement | Overcharged for something you did purchase with the card | 687 | Not fraud (0) |
| Credit card or prepaid card | Getting a credit card | Delay in processing application | 603 | Not fraud (0) |
| Credit card or prepaid card | Problem getting a card or closing an account | Trouble closing card | 596 | Not fraud (0) |
| Credit card or prepaid card | Problem getting a card or closing an account | Trouble getting, activating, or registering a card | 590 | Not fraud (0) |
| Credit card or prepaid card | Other features, terms, or problems | Privacy issues | 555 | Not fraud (0) |
| Credit card or prepaid card | Trouble using the card | Trouble getting information about the card | 447 | Not fraud (0) |
| Credit card or prepaid card | Getting a credit card | Problem getting a working replacement card | 425 | Not fraud (0) |
| Credit card or prepaid card | Other features, terms, or problems | Add-on products and services | 365 | Not fraud (0) |
| Credit card or prepaid card | Improper use of your report | Reporting company used your report improperly | 318 | Not fraud (0) |
| Credit card or prepaid card | Trouble using the card | Problem using the card to withdraw money from an ATM | 317 | Not fraud (0) |
| Credit card or prepaid card | Problem getting a card or closing an account | Trouble getting a working replacement card | 308 | Not fraud (0) |
| Credit card or prepaid card | Problem getting a card or closing an account | Don't want a card provided by your employer or the government | 209 | Not fraud (0) |
| Credit card or prepaid card | Trouble using the card | Problem with direct deposit | 203 | Not fraud (0) |
| Credit card or prepaid card | Problem with a credit reporting company's investigation into an existing problem | Problem with personal statement of dispute | 152 | Not fraud (0) |
| Credit card or prepaid card | Advertising | Confusing or misleading advertising about the card | 140 | Not fraud (0) |
| Credit card or prepaid card | Trouble using the card | Trouble using the card to pay a bill | 137 | Not fraud (0) |
| Credit card or prepaid card | Other features, terms, or problems | Credit card company forcing arbitration | 121 | Not fraud (0) |
| Credit card or prepaid card | Incorrect information on your report | Old information reappears or never goes away | 116 | Not fraud (0) |
| Credit card or prepaid card | Problem with a credit reporting company's investigation into an existing problem | Difficulty submitting a dispute or getting information about a dispute over the phone | 115 | Not fraud (0) |
| Credit card or prepaid card | Other features, terms, or problems | Problem with cash advances | 106 | Not fraud (0) |
| Credit card or prepaid card | Problem with a credit reporting company's investigation into an existing problem | Investigation took more than 30 days | 72 | Not fraud (0) |
| Credit card or prepaid card | Struggling to pay your bill | Problem lowering your monthly payments | 71 | Not fraud (0) |
| Credit card or prepaid card | Incorrect information on your report | Information is missing that should be on the report | 67 | Not fraud (0) |
| Credit card or prepaid card | Problem with overdraft | Overdraft charges | 63 | Not fraud (0) |
| Credit card or prepaid card | Advertising | Changes in terms from what was offered or advertised | 56 | Not fraud (0) |
| Credit card or prepaid card | Trouble using your card | Account sold or transferred to another company | 53 | Not fraud (0) |
| Credit card or prepaid card | Trouble using the card | Problem adding money | 53 | Not fraud (0) |
| Credit card or prepaid card | Incorrect information on your report | Personal information incorrect | 51 | Not fraud (0) |
| Credit card or prepaid card | Problem with a purchase or transfer | Overcharged for a purchase or transfer you did make with the card | 51 | Not fraud (0) |
| Credit card or prepaid card | Struggling to pay your bill | Filed for bankruptcy | 45 | Not fraud (0) |
| Credit card or prepaid card | Unable to get your credit report or credit score | Other problem getting your report or credit score | 43 | Not fraud (0) |
| Credit card or prepaid card | Other features, terms, or problems | Problem with convenience check | 43 | Not fraud (0) |
| Credit card or prepaid card | Trouble using the card | Trouble using the card to send money to another person | 31 | Not fraud (0) |
| Credit card or prepaid card | Incorrect information on your report | Public record information inaccurate | 27 | Not fraud (0) |
| Credit card or prepaid card | Trouble using the card | Problem with a check written from your prepaid card account | 18 | Not fraud (0) |
| Credit card or prepaid card | Improper use of your report | Received unsolicited financial product or insurance offers after opting out | 8 | Not fraud (0) |
| Credit card or prepaid card | Improper use of your report | Report provided to employer without your written authorization | 7 | Not fraud (0) |
| Credit card or prepaid card | Unable to get your credit report or credit score | Problem getting your free annual credit report | 2 | Not fraud (0) |
| Credit card or prepaid card | Problem with an overdraft | Overdraft charges | 2 | Not fraud (0) |
| Credit card or prepaid card | Problem with an overdraft | Was signed up for overdraft on card, but don't want to be | 1 | Not fraud (0) |
| Credit card or prepaid card | Problem with overdraft | Was signed up for overdraft on card, but don't want to be | 1 | Not fraud (0) |
| Debt collection | Attempts to collect debt not owed | Debt is not yours | 36,925 | Fraud (1) |
| Debt collection | Attempts to collect debt not owed | Debt was result of identity theft | 22,562 | Fraud (1) |
| Debt collection | False statements or representation | Impersonated attorney, law enforcement, or government official | 2,135 | Fraud (1) |
| Debt collection | Written notification about debt | Didn't receive enough information to verify debt | 17,990 | Not fraud (0) |
| Debt collection | Attempts to collect debt not owed | Debt was paid | 14,245 | Not fraud (0) |
| Debt collection | False statements or representation | Attempted to collect wrong amount | 12,709 | Not fraud (0) |
| Debt collection | Written notification about debt | Didn't receive notice of right to dispute | 9,846 | Not fraud (0) |
| Debt collection | Took or threatened to take negative or legal action | Threatened or suggested your credit would be damaged | 7,706 | Not fraud (0) |
| Debt collection | Communication tactics | Frequent or repeated calls | 7,369 | Not fraud (0) |
| Debt collection | Communication tactics | You told them to stop contacting you, but they keep trying | 4,304 | Not fraud (0) |
| Debt collection | Took or threatened to take negative or legal action | Threatened to sue you for very old debt | 3,082 | Not fraud (0) |
| Debt collection | Attempts to collect debt not owed | Debt was already discharged in bankruptcy and is no longer owed | 2,849 | Not fraud (0) |
| Debt collection | Threatened to contact someone or share information improperly | Talked to a third-party about your debt | 1,970 | Not fraud (0) |
| Debt collection | Took or threatened to take negative or legal action | Sued you without properly notifying you of lawsuit | 1,795 | Not fraud (0) |
| Debt collection | Communication tactics | Used obscene, profane, or other abusive language | 1,739 | Not fraud (0) |
| Debt collection | Took or threatened to take negative or legal action | Seized or attempted to seize your property | 1,506 | Not fraud (0) |
| Debt collection | False statements or representation | Indicated you were committing crime by not paying debt | 1,161 | Not fraud (0) |
| Debt collection | Written notification about debt | Notification didn't disclose it was an attempt to collect a debt | 1,121 | Not fraud (0) |
| Debt collection | Took or threatened to take negative or legal action | Threatened to arrest you or take you to jail if you do not pay | 985 | Not fraud (0) |
| Debt collection | Took or threatened to take negative or legal action | Collected or attempted to collect exempt funds | 825 | Not fraud (0) |
| Debt collection | Threatened to contact someone or share information improperly | Contacted you after you asked them to stop | 765 | Not fraud (0) |
| Debt collection | Threatened to contact someone or share information improperly | Contacted your employer | 736 | Not fraud (0) |
| Debt collection | Communication tactics | Called before 8am or after 9pm | 608 | Not fraud (0) |
| Debt collection | Took or threatened to take negative or legal action | Sued you in a state where you do not live or did not sign for the debt | 358 | Not fraud (0) |
| Debt collection | False statements or representation | Told you not to respond to a lawsuit they filed against you | 348 | Not fraud (0) |
| Debt collection | Threatened to contact someone or share information improperly | Contacted you instead of your attorney | 61 | Not fraud (0) |
| Debt collection | Took or threatened to take negative or legal action | Threatened to turn you in to immigration or deport you | 19 | Not fraud (0) |
| Debt collection | Electronic communications | You told them to stop contacting you, but they keep trying | 6 | Not fraud (0) |
| Debt collection | Electronic communications | Frequent or repeated messages | 4 | Not fraud (0) |
| Debt collection | Electronic communications | Contacted before 8am or after 9pm | 1 | Not fraud (0) |
| Money transfer, virtual currency, or money service | Fraud or scam | — | 11,326 | Fraud (1) |
| Money transfer, virtual currency, or money service | Unauthorized transactions or other transaction problem | — | 2,990 | Fraud (1) |
| Money transfer, virtual currency, or money service | Lost or stolen check | — | 226 | Excluded |
| Money transfer, virtual currency, or money service | Lost or stolen money order | — | 105 | Excluded |
| Money transfer, virtual currency, or money service | Other transaction problem | — | 4,945 | Not fraud (0) |
| Money transfer, virtual currency, or money service | Managing, opening, or closing your mobile wallet account | — | 3,578 | Not fraud (0) |
| Money transfer, virtual currency, or money service | Money was not available when promised | — | 2,834 | Not fraud (0) |
| Money transfer, virtual currency, or money service | Other service problem | — | 1,109 | Not fraud (0) |
| Money transfer, virtual currency, or money service | Confusing or missing disclosures | — | 906 | Not fraud (0) |
| Money transfer, virtual currency, or money service | Unexpected or other fees | — | 717 | Not fraud (0) |
| Money transfer, virtual currency, or money service | Problem with customer service | — | 651 | Not fraud (0) |
| Money transfer, virtual currency, or money service | Wrong amount charged or received | — | 413 | Not fraud (0) |
| Money transfer, virtual currency, or money service | Problem adding money | — | 361 | Not fraud (0) |
| Money transfer, virtual currency, or money service | Confusing or misleading advertising or marketing | — | 323 | Not fraud (0) |
| Money transfer, virtual currency, or money service | Overdraft, savings, or rewards features | — | 51 | Not fraud (0) |
| Money transfer, virtual currency, or money service | Incorrect exchange rate | — | 46 | Not fraud (0) |
| Money transfer, virtual currency, or money service | Trouble accessing funds in your mobile or digital wallet | — | 8 | Not fraud (0) |
| Prepaid card | Problem with a purchase or transfer | Charged for a purchase or transfer you did not make with the card | 8 | Fraud (1) |
| Prepaid card | Problem with a purchase or transfer | Card company isn't resolving a dispute about a purchase or transfer | 5 | Excluded |
| Prepaid card | Trouble using the card | Trouble using the card to spend money in a store or online | 13 | Not fraud (0) |
| Prepaid card | Unexpected or other fees | — | 11 | Not fraud (0) |
| Prepaid card | Advertising | Confusing or misleading advertising about the card | 3 | Not fraud (0) |
| Prepaid card | Problem getting a card or closing an account | Trouble closing card | 2 | Not fraud (0) |
| Prepaid card | Problem getting a card or closing an account | Trouble getting, activating, or registering a card | 2 | Not fraud (0) |
| Prepaid card | Problem getting a card or closing an account | Trouble getting a working replacement card | 2 | Not fraud (0) |
| Prepaid card | Trouble using the card | Trouble getting information about the card | 1 | Not fraud (0) |
| Prepaid card | Trouble using the card | Trouble using the card to send money to another person | 1 | Not fraud (0) |
