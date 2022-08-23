# Entry point for the backtester.

from quote import Quote
from ml_strategy import Ml_Strategy
from trade_situation import TradeSituation
from limit_order_book import LimitOrderBook
from curr_pair import CurrPair
from new_cancel import NewCancel
import numpy as np
import os

# Some static functions
def count_rows_in_file(file_name: str) -> int:
    row_count_local: int = 0
    with open(file_name, 'r') as file_reader:
        chunk = file_reader.readline()
        while chunk != '':
            row_count_local += 1
            chunk = file_reader.readline()
    return row_count_local


# Filename
path = os. getcwd()
data_file_name = path+ "/livefix-log-18Jan-09-52-10-774.csv"

# count lines
row_count: float = count_rows_in_file(data_file_name)
print("Total {0} rows in {1}.".format(row_count, data_file_name))

# Create a placeholder for all the data:
quotes: tuple = (None,) * row_count

# Create an instance of ML Strategy class
# THE AMOUNT IS USED FOR PRICE REFERENCE!
traded_amount = 300000.00
target_profit = 0.00003
curr_pair = CurrPair.EURUSD
lookback=500
strategy = Ml_Strategy(lookback, target_profit, traded_amount, True)
limit_order_book = LimitOrderBook(curr_pair)

Ml_Strategy.set_limit_order_book(limit_order_book)
TradeSituation.set_limit_order_book(limit_order_book)
level=5
best_offer_best_bid=None


# FOR loop on quotes: create the Quote instance objects (one per quote line) and feed it (step()) to the strategy object
lines_read_so_far: float = 0.0
with open(data_file_name, 'r') as reader_obj:
    while lines_read_so_far < row_count:
        row = reader_obj.readline()
        # recognize the row's contents
        quote = Quote(row)
        if quote.currency_pair() == curr_pair:
            # Update order book
            if quote.type() == NewCancel.NEW:
                limit_order_book.on_new_order(quote)
                # Update strategy: by construction this ECN sends an update to the price immediately.
                # The cancels are ignored for the strategy updates.
                if len(limit_order_book.limit_bids())>level and len(limit_order_book.limit_offers())>level:
                    
                    list_best_bids=[]
                    for keys,value in limit_order_book.limit_bids().items():

                        list_best_bids.append(keys)
                        list_best_bids.append(limit_order_book.limit_bids()[keys][-1].amount())
                                

                    list_best_offers=[]
                    for keys,value in limit_order_book.limit_offers().items():

                        list_best_offers.append(keys)
                        list_best_offers.append(limit_order_book.limit_offers()[keys][-1].amount())
                    
                    best_row=np.array(list_best_offers[0:2*level]+list_best_bids[0:2*level])
                    if best_offer_best_bid is None:
                        best_offer_best_bid=best_row.reshape(1,20)
                    else:
                        if not (best_row==best_offer_best_bid[-1]).all():
                            best_offer_best_bid=np.vstack([best_offer_best_bid,best_row])
                    
                #prediction et strategy


                strategy.step(quote,best_offer_best_bid)

                if best_offer_best_bid is not None:
                    if len(best_offer_best_bid)>=lookback:
                        print('clean')
                        #initialisation de best_offer_best_bid et order_book_array
                        best_offer_best_bid=None
                        #limit_order_book.initialize_order_book_array()
                
            else:
                limit_order_book.on_cancel_order(quote)
        # Update user interface statistics
        lines_read_so_far += 1.0
        # Output statistics to user:
        if lines_read_so_far % 1000 == 0:
            print("Read {0} lines ({1:2.2f}%)".format(lines_read_so_far, 100.00 * lines_read_so_far / row_count))

# Close remaining position to output trade statistics
strategy.close_pending_position(quote)
all_predict_probability=strategy.get_all_predict_probability()

total_pnl: float = 0.0
maximal_draw_down: float = 0.0
position: TradeSituation
for position in strategy.all_positions():
    total_pnl += position.return_current_pnl()
    if position.return_current_draw_down() > maximal_draw_down:
        maximal_draw_down = position.return_current_draw_down()
    print("Position {0}: profit bps {1:2.6f}, draw down bps {2:2.6f}".format(position.trade_situation_id(),
                                                                             position.return_current_pnl(),
                                                                             position.return_current_draw_down()))

positions_opened: int = len(strategy.all_positions())
print("Total {0} positions opened.".format(positions_opened))
print("Total profit (loss) in basis points is: {0:2.2f}.".format(total_pnl))
print("Maximal draw down in basis points is: {0:2.6f}.".format(maximal_draw_down))
print("Calmar ratio: {0:2.6f}.".format(total_pnl/maximal_draw_down))


# Transaction price per million USD:
price_per_mil = 10.00
# We buy then we sell => thus it's traded amount X 2
transaction_price_per_trade = traded_amount * 2.0 * price_per_mil / 1000000.00
print("Total transaction price: {0:2.2f}.".format(transaction_price_per_trade * positions_opened))
print("Total profit (loss): {0:2.2f}.".format(traded_amount * total_pnl))
print("Net profit (loss): {0:2.2f}.".format(traded_amount * total_pnl - transaction_price_per_trade * positions_opened))


