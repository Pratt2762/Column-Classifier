import pandas as pd

# Adding some new data points to our dataset to increase its diversity
# The added data tries to imitate the data distribution of the given dataset
def add_synthetic_data(dataset):
    added_data_interest_rate = {"text":["3.2%", "5.6%", "low", "low", "0.04", "0.034", "4.332%", "low", "high", "high",
                                       "medium", "medium", "5%", "low", "lowest"],
                               "tag":["interest_rate" for _ in range(15)]}

    df = pd.DataFrame(added_data_interest_rate)
    dataset = pd.concat([dataset, df])
    dataset = pd.concat([dataset, df])
    dataset = pd.concat([dataset, df])

    added_data_income_yearly = {"text":["34000", "260,000", "median", "median", "278000", "income", "below_min_wage", "income",
                                        "below_min_wage", "25799", "median", "98000", "20,000.00", "below_min_wage", "56,000"],
                               "tag":["income_yearly" for _ in range(15)]}

    df = pd.DataFrame(added_data_income_yearly)
    dataset = pd.concat([dataset, df])
    dataset = pd.concat([dataset, df])
    dataset = pd.concat([dataset, df])


    added_data_customer_satisfaction = {"text":["unhappy", "satisfied", "satisfied", "happy", "okay", "okay", "unsatisfied", 
                                               "neutral", "very happy", "good", "satisfactory", "very satisfied", "poor",
                                                "poor", "great"], 
                                       "tag":["customer_satisfaction" for _ in range(15)]}

    df = pd.DataFrame(added_data_customer_satisfaction)
    dataset = pd.concat([dataset, df])
    dataset = pd.concat([dataset, df])
    dataset = pd.concat([dataset, df])
    
    return dataset
