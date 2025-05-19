from typing import Iterator, Union

class Query:
    def components(self) -> Iterator['Query']:
        """
        Return an iterator over the components of the query.
        For example, A AND B AND C would return A, B and C.
        A as an atomic query would return itself.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __len__(self) -> int:
        """
        Return the number of components in the query.
        """
        return sum(1 for _ in self.components())
    
    def __iter__(self) -> Iterator['Query']:
        """
        Return an iterator over the components of the query.
        """
        return self.components()
    
    def __eq__(self, other: 'Query') -> bool:
        """
        Check if two queries are equal.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __hash__(self) -> int:
        """
        Return a hash of the query.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __repr__(self) -> str:
        """
        Return a string representation of the query.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __str__(self) -> str:
        """
        Return a string representation of the query.
        """
        return self.__repr__()
