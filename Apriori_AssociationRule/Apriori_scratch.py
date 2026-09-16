import pandas as pd
from itertools import combinations

df = pd.read_csv("Apriori_AssociationRule/Groceries_dataset.csv")
df["Transaction"] = (
    df["Member_number"].astype(str)
    + "_"
    + df["Date"].astype(str)
)

transactions = df.groupby("Transaction")["itemDescription"].apply(set).tolist()
print("Total Transactions:", len(transactions))

def calculate_support(itemset,transactions):
    count=0
    for transaction in transactions:
        if itemset.issubset(transaction):
            count+=1
    return count/len(transactions)

def generate_candidates(previous_frequent, size):
    candidates=set()
    previous_frequent=list(previous_frequent)
    for i in range(len(previous_frequent)):
        for j in range(i + 1, len(previous_frequent)):
            union = previous_frequent[i] | previous_frequent[j]
            if len(union) == size:
                candidates.add(frozenset(union))
    return candidates  

def apriori(transactions, min_support):
    all_items = set()
    for transaction in transactions:
        for item in transaction:
            all_items.add(item)

    candidates=set()
    for item in all_items:
        candidates.add(frozenset([item]))
    frequent_itemsets = {}
    
    current_frequent = set()
    
    for itemset in candidates:
    
        support = calculate_support(
            itemset,
            transactions
        )

        if support >= min_support:

            current_frequent.add(itemset)
            frequent_itemsets[itemset] = support

    k = 2   
    while current_frequent:
            candidates = generate_candidates(
                current_frequent,
                k
            )
            next_frequent = set()
            for itemset in candidates:
                support = calculate_support(
                    itemset,
                    transactions
                )
                if support >= min_support:
                    next_frequent.add(itemset)
                    frequent_itemsets[itemset] = support
            current_frequent = next_frequent
            k += 1
    return frequent_itemsets
    

min_support = 0.001

frequent_itemsets = apriori(
    transactions,
    min_support
)

print(
    "\nTotal Frequent Itemsets:",
    len(frequent_itemsets)
)


def generate_rules(frequent_itemsets, transactions, min_confidence):
    rules = []
    for itemset, itemset_support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)

        for size in range(1, len(items)):
            for antecedent_items in combinations(items, size):
                antecedent = frozenset(antecedent_items)
                consequent = itemset - antecedent
                antecedent_support = frequent_itemsets.get(
                    antecedent,
                    0
                )

                if antecedent_support == 0:
                    continue
                confidence = (
                    itemset_support
                    / antecedent_support
                )
                if confidence >= min_confidence:
                    consequent_support = calculate_support(
                        consequent,
                        transactions
                    )
                    lift = (
                        confidence
                        / consequent_support
                    )
                    rules.append(
                        (
                            antecedent,
                            consequent,
                            itemset_support,
                            confidence,
                            lift
                        )
                    )
    return rules

min_confidence = 0.10
rules = generate_rules(
    frequent_itemsets,
    transactions,
    min_confidence
)
rules.sort(
    key=lambda x: x[4],
    reverse=True
)

print("---- STRONGEST RULES ----")

for antecedent, consequent, support, confidence, lift in rules[:20]:

    print(
        f"{', '.join(antecedent)}"
        f" --> "
        f"{', '.join(consequent)}"
    )
    print(
        f"Support: {support:.3f} | "
        f"Confidence: {confidence:.3f} | "
        f"Lift: {lift:.3f}"
    )
    print()

query = "whole milk"

print("\n========== ASSOCIATIONS FOR", query, "==========")
for antecedent, consequent, support, confidence, lift in rules:
    antecedent_lower = {
        item.lower()
        for item in antecedent
    }

    if query.lower() in antecedent_lower:

        print(
            f"{', '.join(antecedent)}"
            f" --> "
            f"{', '.join(consequent)}"
        )
        print(
            f"Support: {support:.3f} | "
            f"Confidence: {confidence:.3f} | "
            f"Lift: {lift:.3f}"
        )

        print()