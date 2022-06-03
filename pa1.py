from apyori import apriori
import pandas as pd

lines = []
doubles = []
triples = []
with open("browsing-data.txt") as data:
    for line in data:
        lines.append(line.split())
#using documentation from: https://zaxrosenberg.com/unofficial-apyori-documentation/ as a guide since offical library has AWFUL documentation
rules = apriori(
    lines,
    min_support=100/len(lines),
    min_confidence=0.95,
    min_lift=3,
    max_length=3
)
results=list(rules)
for rule in results:
    things = [i for i in rule[0]]
    if len(things) == 2:
        doubles.append([things[0], things[1], str(rule[2][0][2])])
    elif len(things) == 3:
        triples.append([things[0], things[1], things[2], str(rule[2][0][2])])

df = pd.DataFrame(doubles, columns=["item1", "item2", "confidence"])
df = df.sort_values(by='confidence', ascending=False)

df2 = pd.DataFrame(triples, columns=["item1", "item2", "item3", "confidence"])
df2 = df2.sort_values(by='confidence', ascending=False)

#print heads of dataframes (top 5 results) into output.txt
with open("output.txt", "w+") as dataout:
    dataout.write("Output A\n")
    dataout.write(df.head().to_string())
    dataout.write("\nOutput B\n")
    dataout.write(df2.head().to_string())