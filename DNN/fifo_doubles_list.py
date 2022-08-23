# This class helps manage your fixed list of values.


class FifoDoublesList:
    # The index in the array to be updated next.
    __next_updated_index: int
    # Length of the array
    __list_length_local: int

    # List containing all the available data (float)
    __data_list: list

    def __init__(self, list_length: int):
        """
        Creates a ?higher? performance ...dynamic... array with values overwritten on each call
        :param list_length: length of the created array
        """
        # Sanity check: list must be > 1
        if list_length <= 1:
            raise Exception("Please init the class with a length of an array higher than 1")
        # Initialize __list_length_local and __data_list (to an array of 0.00)
        self.__list_length_local = list_length
        self.__data_list = [0.00] * self.__list_length_local
        # Init locals
        self.__next_updated_index = 0

    def put(self, inserted_value: float):
        """
        Adds the argument into the array (overwriting the oldest value ion that array)
        :param inserted_value: the value that will be inserted
        :return: no return
        """
        # Insert the argument's value into __data_list. Update index variable accordingly.
        self.__data_list[self.__next_updated_index] = inserted_value
        self.__next_updated_index += 1
        # We've hit the last index of the array. Reset the index to 0.
        if self.__next_updated_index == self.__list_length_local:
            self.__next_updated_index = 0

    def get_sum(self) -> float:
        """
        This returns the sum of the list
        :return: float value; sum of the created array
        """
        # Calculate the sum of the __data_list
        list_sum = sum(self.__data_list)
        return list_sum

    def get_mean(self) -> float:
        """
        This returns the mean of the list
        :return: float value; sum of the created array
        """
        # Calculate the mean of __data_list's values
        array_sum = sum(self.__data_list)
        return array_sum / self.__list_length_local

    def size(self) -> int:
        """
        Gets the size of this array
        :return: int; size of the array
        """
        # Return the length of the list.
        return self.__list_length_local

    def return_values(self) -> list:
        """
        Returns an ordered list of values of this array. The first value (at index 0) is the one that was added first
        in the collection (oldest).
        :return: collection ordered by time of arrival.
        """
        # Init the returned list
        returned_list = [None] * self.__list_length_local
        # Init the index that is used for the __data_list data fetch
        moved_last_index = self.__next_updated_index
        # FOR on all of the __data_list length
        for index_value in range(self.__list_length_local):
            # Add values to the returned list (from __data_list)
            returned_list[index_value] = self.__data_list[moved_last_index]
            # Check the index used in the __data_list
            moved_last_index += 1
            if moved_last_index == self.__list_length_local:
                moved_last_index = 0
        return returned_list
