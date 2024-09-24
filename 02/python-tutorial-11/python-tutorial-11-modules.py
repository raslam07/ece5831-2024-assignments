from pricing import get_net_price as calculate_net_price

net_price = calculate_net_price(
    price=100,
    tax_rate=0.1,
    discount=0.05
)

print(net_price)

from product import *

tax = get_tax(100)
print(tax)