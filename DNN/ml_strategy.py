from fifo_doubles_list import FifoDoublesList
from quote import Quote
from trade_situation import TradeSituation
from limit_order_book import LimitOrderBook
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
from buy_sell import BuySell


class Ml_Strategy:
    # This variable will be incremented after each call of TradeSituation.generate_next_id().
    # It is used to populate __trade_situation_id.
    __common_momentum_strategy_id: int = 0
    # This is a global reference to the order book
    __common_order_book: LimitOrderBook
    # Unique ID of the Ml strategy
    __strategy_id: int
    # Set to final value in constructor.
    # Predicted value of the trend variable
    __predict_probability:float
    # All predicted values
    __all_predict_probability:list
    # Number of data points used for the model construction
    __lookback: int
    # DNN Model
    __dnn_model: tf.keras.Sequential
    # Position is open in this way currently
    # False: sold
    # True: bought
    __current_trading_way: bool

    # Currently opened position
    __open_position: TradeSituation

    # List of trade situation
    __positions_history: list

    # This variable is set once to True when the required (minimal) data points are populated into fifo_list(s)
    __is_filled_start_data: bool
    __filled_data_points: int
    # This variable describes that we are using the best BID and best OFFER to calculate PnL
    __is_best_price_calculation: bool
    # This is the strategy's traded amount
    __traded_amount: float


    def __init__(self,lookback: int, target_profit_arg: float, traded_amount: float, is_best_px_calc: bool):
        """
        Initializes the trading strategy calculator. Please feed it with arguments for your moving average trading
        strategy. The MA_SLOW > MA_FAST. By construction the FAST average is low-period.
        :param ma_slow: slow moving moving average
        :param ma_fast: fast moving moving average
        :param target_profit_arg: target profit for this strategy
        """
        self.__lookback=lookback
        #self.__proba_down = proba_down
        #self.__proba_up = proba_up
        self.__strategy_id = Ml_Strategy.generate_next_id()
        self.__is_best_price_calculation = is_best_px_calc
        self.__traded_amount = traded_amount
        self.__dnn_model=None
        self.__predict_probability = 0.0
        self.__all_predict_probability=[]
        # Save input arguments

        self.__target_profit = target_profit_arg


        # Init locals
        self.__current_trading_way = False
        self.__open_position = None
        self.__positions_history = []
        self.__is_filled_start_data = False
        self.__filled_data_points = 0

    def step(self, quote: Quote,best_offer_best_bid):
        """
        Calculates the indicator and performs update/open/close action on the TradeSituation class
        (representing investment position)
        :param quote: float; the price of the invested stock
        :return: no return
        """
        # Update values (prices) in the fifo_lists (with put method)
        
        price_mid: float = (self.__common_order_book.get_best_bid_price() +\
                            self.__common_order_book.get_best_offer_price()) / 2.0
        
        # Update position with arrived quote
        if self.__open_position is not None:
            # We closed the position (returns true if the position is closed)
            if self.__open_position.update_on_order(quote):
                self.__open_position = None

        self.__filled_data_points += 1
        # >lookback
        if self.__is_filled_start_data:
            
            
            if best_offer_best_bid is not None:
              if len(best_offer_best_bid)>=self.__lookback:
                  
                  #Prediction
                  if self.__dnn_model is not None: 
                      X_test=best_offer_best_bid[-1].reshape(1,20)
                      self.__predict_probability=self.__dnn_model.predict(X_test)[0,0]
                      self.__all_predict_probability.append(self.__predict_probability)
                  #Model construction 
                  self.model_construction(best_offer_best_bid)
                  
              else:
                  #Prediction
                  X_test=best_offer_best_bid[-1].reshape(1,20)
                  self.__predict_probability= self.__dnn_model.predict(X_test)[0,0]
                  self.__all_predict_probability.append(self.__predict_probability)

            # You must not reopen the position if the trading direction (__current_trading_way) has not changed.
            if self.compare_midprice_quote(price_mid,quote)==BuySell.BUY and not self.__current_trading_way :
                # Buy: open position if there is none; close the position if it's hanging in the other way; append the
                # positions history (to save how much it gained); save the new __current_trading_way (repeat for SELL)
                if self.__open_position is not None:
                    self.__open_position.close_position(quote)
                self.__open_position = TradeSituation(quote, True, self.__target_profit, self.__traded_amount,
                                                      self.__is_best_price_calculation)
                self.__open_position.open_position(quote)
                self.__current_trading_way = True
                self.__positions_history.append(self.__open_position)
            elif self.compare_midprice_quote(price_mid,quote)==BuySell.SELL and self.__current_trading_way:
                # Sell
                if self.__open_position is not None:
                    self.__open_position.close_position(quote)
                self.__open_position = TradeSituation(quote, False, self.__target_profit, self.__traded_amount,
                                                      self.__is_best_price_calculation)
                self.__current_trading_way = False
                self.__positions_history.append(self.__open_position)
        else:
            # The fifo_list(s) are not yet filled. Do the necessary updates and checks
            if best_offer_best_bid is not None:
                if len(best_offer_best_bid)>= self.__lookback:
                    self.__is_filled_start_data = True
                    #Model construction 
                    print('init')
                    self.model_construction(best_offer_best_bid)
        
    
    def close_pending_position(self, quote: Quote):
        """
        Called at the end of the program execution. Checks if the position is still opened and closes that position.
        :param quote: last quote available in the data set
        :return:
        """
        # If there is still a position --> close it with the quote provided to you in arguments.
        if self.__open_position is not None and not self.__open_position.is_closed():
            self.__open_position.close_position(quote)

    def all_positions(self) -> list:
        """
        Returns the positions_history object
        :return:
        """
        # Returns __positions_history
        return self.__positions_history

    def get_lookback(self) -> int:
        """
        Returns the value of lookback
        :return:
        """
        return self.__lookback


    def get_target_profit(self) -> float:
        """
        Returns the target profit of this strategy
        :return:
        """
        return self.__target_profit

    def get_strategy_id(self):
        """
        Returns the (local) unique strategy ID.
        :return:
        """
        return self.__strategy_id
    
    def get_all_predict_probability(self):
        """
        Returns all_predict_probability
        :return:
        """
        return self.__all_predict_probability
    
    def compare_midprice_quote(self,midPrice:float, quote:Quote):
        """
        Returns the trading way.
        :return:
        """
        trading_way = None
        #around 1
        if self.__predict_probability < 1.02 and self.__predict_probability >0.98 and quote.price() <= midPrice:
            trading_way=BuySell.BUY
        #around 3
        elif self.__predict_probability >2.98 and self.__predict_probability < 3.02 and quote.price() < midPrice:
            trading_way=BuySell.BUY
        #around 2
        elif self.__predict_probability>=1.98 and self.__predict_probability<=2.02 and quote.price() >= midPrice:
            trading_way = BuySell.SELL
        #around 3
        elif self.__predict_probability >2.98 and self.__predict_probability < 3.02 and quote.price() > midPrice:
            trading_way = BuySell.SELL

        return trading_way
    def model_construction(self,best_offer_best_bid):
        """
        Construct the DNN model
        """
        #Train variables
        X_train = best_offer_best_bid[0:int(self.__lookback/2),:]
        data_y=best_offer_best_bid[int(self.__lookback/2):]
        Y_train= Ml_Strategy.compute_prob(X_train,data_y)
        #Initialisation/Mise à jour du modèle
        normalizer = preprocessing.Normalization()
        normalizer.adapt(np.array(X_train))
        self.__dnn_model = Ml_Strategy.dnn_classification(normalizer)
        dnnhistory = self.__dnn_model.fit(
            X_train, Y_train,
            validation_split=0.2,
            verbose=0, epochs=100)
        print(str(self.__filled_data_points))
        print(str(len(best_offer_best_bid)))

    @staticmethod
    def generate_next_id():
        Ml_Strategy.__common_momentum_strategy_id += 1
        return Ml_Strategy.__common_momentum_strategy_id

    @staticmethod
    def set_limit_order_book(limit_order_book: LimitOrderBook):
        Ml_Strategy.__common_order_book = limit_order_book
    
    @staticmethod
    def dnn_classification(norm):
        """
        DNN model definition
        Returns the DNN model
        """
        model = keras.Sequential([
            norm,
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)])
        model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    @staticmethod
    def compute_prob(data_x,data_y):
        """
        Returns the trend variable
        data_x: the first half of dataset
        data_y: the second hald of dataset
        """
        midPrice=(data_x[:,0] + data_x[:,10])/2
        n=len(data_x)
        midAverage = (data_y[:,0] + data_y[:,10])/2
        alpha = 0.00002
        prob = np.zeros([n, 1])
        for i in range(n):
            #proba que mid va diminuer
            if midPrice[i]*(1+alpha) < midAverage[i]:
                prob[i][0] = 1
            #proba que mid va augmenter
            elif midPrice[i]*(1-alpha) > midAverage[i]:
                prob[i][0] = 2
            #proba que mid va stagner
            else:
                prob[i][0] = 3

        return prob
