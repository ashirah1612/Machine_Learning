import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv(
    "Apriori_AssociationRule/Groceries_dataset.csv"
)


df["Transaction"] = (
    df["Member_number"].astype(str)
    + "_"
    + df["Date"].astype(str)
)


transactions = (
    df.groupby("Transaction")["itemDescription"]
    .apply(list)
    .tolist()
)


encoder = TransactionEncoder()

encoded_data = encoder.fit(transactions).transform(transactions)
basket=pd.DataFrame(encoded_data,columns=encoder.columns_)
frequent_itemsets=apriori(basket,min_support=0.001,use_colnames=True)
print("Total Transactions:",len(transactions))
print("Total Frequent Itemsets:",len(frequent_itemsets))

rules=association_rules(frequent_itemsets,metric="confidence",min_threshold=0.10)
rules=rules.sort_values(by="lift",ascending=False)


print("\n========== STRONGEST RULES ==========")

for _, rule in rules.head(20).iterrows():
    print(
        f"{', '.join(rule['antecedents'])}"
        f" --> "
        f"{', '.join(rule['consequents'])}"
    )
    print(
        f"Support: {rule['support']:.3f} | "
        f"Confidence: {rule['confidence']:.3f} | "
        f"Lift: {rule['lift']:.3f}"
    )
    print()
query = "whole milk"


print("\n========== ASSOCIATIONS FOR",query,"==========")


for _, rule in rules.iterrows():
    antecedent = {
        item.lower()
        for item in rule["antecedents"]
    }
    if query.lower() in antecedent:
        print(
            f"{', '.join(rule['antecedents'])}"
            f" --> "
            f"{', '.join(rule['consequents'])}"
        )
        print(
            f"Support: {rule['support']:.3f} | "
            f"Confidence: {rule['confidence']:.3f} | "
            f"Lift: {rule['lift']:.3f}"
        )
        print()