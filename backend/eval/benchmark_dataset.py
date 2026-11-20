"""Curated 40 Evaluation Benchmark Questions for pythonlearn.pdf (Python for Everybody).

Covers 10 distinct knowledge categories:
1. Basic factual questions
2. Semantic search and paraphrased questions
3. Questions requiring information from multiple sections
4. Python code examples and syntax
5. Lists, dictionaries, and tuples
6. Regular expressions
7. Networking and web services (HTTP/sockets)
8. JSON, XML, and data formats
9. Database and SQL (SQLite)
10. Figures, diagrams, and structure
"""

from typing import List, Dict, Any

BENCHMARK_QUESTIONS: List[Dict[str, Any]] = [
    # Category 1: Basic factual questions
    {
        "id": 1,
        "category": "Basic factual questions",
        "query": "Who is the author of Python for Everybody and what was the book formerly titled?",
        "target_pages": ["1", "i", "ii", "iii"],
        "keywords": ["Charles Severance", "Python for Informatics", "author"],
    },
    {
        "id": 2,
        "category": "Basic factual questions",
        "query": "What is the difference between an interpreter and a compiler in Python?",
        "target_pages": ["7", "8", "9", "13", "14", "15", "20", "21"],
        "keywords": ["interpreter", "compiler", "source code", "machine code"],
    },
    {
        "id": 3,
        "category": "Basic factual questions",
        "query": "What are the three general types of programming errors described in the book?",
        "target_pages": ["13", "14", "15", "16", "19", "20", "25", "26"],
        "keywords": ["syntax errors", "runtime errors", "semantic errors"],
    },
    {
        "id": 4,
        "category": "Basic factual questions",
        "query": "What are the rules for legal Python variable names and what characters are allowed?",
        "target_pages": ["21", "22", "27", "33"],
        "keywords": ["variable names", "letters", "numbers", "underscore", "illegal"],
    },
    {
        "id": 5,
        "category": "Basic factual questions",
        "query": "What are the reserved keywords in Python that cannot be used as variable names?",
        "target_pages": ["22", "28", "34"],
        "keywords": ["reserved words", "keywords", "and", "del", "for", "is", "raise"],
    },

    # Category 2: Semantic search and paraphrased questions
    {
        "id": 6,
        "category": "Semantic search & paraphrased",
        "query": "How can you prevent a program from crashing when converting user text into a number?",
        "target_pages": ["36", "37", "42", "43", "48", "49"],
        "keywords": ["try and except", "try", "except", "ValueError", "catch"],
    },
    {
        "id": 7,
        "category": "Semantic search & paraphrased",
        "query": "How does Python evaluate compound boolean conditions without evaluating every term?",
        "target_pages": ["38", "39", "44"],
        "keywords": ["short-circuit", "guardian pattern", "short-circuit evaluation"],
    },
    {
        "id": 8,
        "category": "Semantic search & paraphrased",
        "query": "What is the idiom used to safely look through a file line by line without running out of memory?",
        "target_pages": ["80", "81", "86"],
        "keywords": ["file handle", "for line in fhand", "open", "buffer"],
    },
    {
        "id": 9,
        "category": "Semantic search & paraphrased",
        "query": "Why should you pass arguments into functions rather than relying on global state?",
        "target_pages": ["49", "50", "55"],
        "keywords": ["arguments", "parameters", "fruitful functions", "void functions"],
    },

    # Category 3: Questions requiring information from multiple sections
    {
        "id": 10,
        "category": "Multi-section synthesis",
        "query": "How do lists and dictionaries differ in indexing, mutability, and internal lookup implementation?",
        "target_pages": ["91", "109", "110", "116"],
        "keywords": ["dictionary", "list", "hash table", "linear search", "key-value"],
    },
    {
        "id": 11,
        "category": "Multi-section synthesis",
        "query": "How do you combine files, dictionaries, and tuples to compute the most frequent words in a document?",
        "target_pages": ["122", "123", "124", "125", "126"],
        "keywords": ["histogram", "sort by value", "tuples", "reverse=True", "most common words"],
    },
    {
        "id": 12,
        "category": "Multi-section synthesis",
        "query": "Compare the guardian pattern used in while loops with defensive programming using try-except in file handling.",
        "target_pages": ["38", "86", "87"],
        "keywords": ["guardian", "try", "except", "defensive"],
    },
    {
        "id": 13,
        "category": "Multi-section synthesis",
        "query": "How does data representation evolve from strings to lists, then to dictionaries, and finally to relational tables?",
        "target_pages": ["91", "109", "183", "184"],
        "keywords": ["strings", "lists", "dictionaries", "database", "tables", "relational"],
    },

    # Category 4: Python code examples and syntax
    {
        "id": 14,
        "category": "Python code & syntax",
        "query": "How do you define a function that takes arguments and returns a value using def and return?",
        "target_pages": ["50", "51", "56"],
        "keywords": ["def", "return", "fruitful function"],
    },
    {
        "id": 15,
        "category": "Python code & syntax",
        "query": "Show the syntax for string slicing with start, stop, and step parameters in Python.",
        "target_pages": ["68", "69", "74"],
        "keywords": ["slice", "fruit[0:5]", "index", "colon"],
    },
    {
        "id": 16,
        "category": "Python code & syntax",
        "query": "How do you open a file for reading and strip the trailing newline from each line?",
        "target_pages": ["81", "82", "87"],
        "keywords": ["rstrip", "open", "fhand", "newline"],
    },
    {
        "id": 17,
        "category": "Python code & syntax",
        "query": "What is the format operator '%' and how does string formatting and f-strings work?",
        "target_pages": ["74", "75", "80", "86", "87"],
        "keywords": ["format operator", "%d", "%s", "format sequence", "f-string", "format"],
    },

    # Category 5: Lists, dictionaries, and tuples
    {
        "id": 18,
        "category": "Lists, dicts & tuples",
        "query": "How do you use the get method on dictionaries to build a word frequency histogram?",
        "target_pages": ["111", "112", "118"],
        "keywords": ["counts.get", "histogram", "default value", "dictionary"],
    },
    {
        "id": 19,
        "category": "Lists, dicts & tuples",
        "query": "What is tuple assignment in Python and how can it be used to swap two variables without a temporary variable?",
        "target_pages": ["120", "121", "127"],
        "keywords": ["tuple assignment", "a, b = b, a", "swap"],
    },
    {
        "id": 20,
        "category": "Lists, dicts & tuples",
        "query": "Can lists or tuples be used as keys in a Python dictionary and why?",
        "target_pages": ["126", "127", "133"],
        "keywords": ["hashable", "immutable", "tuples as keys", "TypeError: unhashable type"],
    },
    {
        "id": 21,
        "category": "Lists, dicts & tuples",
        "query": "What is list aliasing and what is the difference between modifying an object and creating a new reference?",
        "target_pages": ["99", "100", "101", "105", "106", "111", "112", "113"],
        "keywords": ["aliasing", "is operator", "identical", "equivalent", "reference"],
    },
    {
        "id": 22,
        "category": "Lists, dicts & tuples",
        "query": "How does the split method convert a string into a list of words?",
        "target_pages": ["97", "98", "103"],
        "keywords": ["split", "delimiter", "words"],
    },
    {
        "id": 23,
        "category": "Lists, dicts & tuples",
        "query": "What is the difference between sorting with the sort() method and the sorted() function?",
        "target_pages": ["95", "123", "124"],
        "keywords": ["sort", "sorted", "reverse", "in-place"],
    },

    # Category 6: Regular expressions
    {
        "id": 24,
        "category": "Regular expressions",
        "query": "How does re.search differ from re.findall in the Python re module?",
        "target_pages": ["130", "131", "133", "137"],
        "keywords": ["re.search", "re.findall", "matches", "extract"],
    },
    {
        "id": 25,
        "category": "Regular expressions",
        "query": "What do the caret '^' and dollar sign '$' characters represent in regular expressions?",
        "target_pages": ["130", "131", "136"],
        "keywords": ["caret", "dollar sign", "start of line", "end of line"],
    },
    {
        "id": 26,
        "category": "Regular expressions",
        "query": "How does greedy vs non-greedy matching work with the '?' modifier in regular expressions?",
        "target_pages": ["134", "135", "140"],
        "keywords": ["greedy", "non-greedy", "*?", "+?", "wild card"],
    },
    {
        "id": 27,
        "category": "Regular expressions",
        "query": "How do parentheses '()' in regular expressions specify extraction targets when using findall?",
        "target_pages": ["135", "136", "141"],
        "keywords": ["parentheses", "extract", "findall", "submatch"],
    },

    # Category 7: Networking and HTTP/sockets
    {
        "id": 28,
        "category": "Networking & web",
        "query": "How do you create a basic HTTP client in Python using the socket module?",
        "target_pages": ["147", "148", "153"],
        "keywords": ["socket", "connect", "send", "recv", "HTTP/1.0", "GET"],
    },
    {
        "id": 29,
        "category": "Networking & web",
        "query": "What is the urllib library in Python and how does it make web requests simpler than raw sockets?",
        "target_pages": ["150", "151", "156"],
        "keywords": ["urllib.request", "urlopen", "treat web page like a file"],
    },
    {
        "id": 30,
        "category": "Networking & web",
        "query": "What is web scraping and how is BeautifulSoup used to parse HTML anchor tags?",
        "target_pages": ["152", "153", "158"],
        "keywords": ["web scraping", "BeautifulSoup", "anchor tags", "href"],
    },
    {
        "id": 31,
        "category": "Networking & web",
        "query": "What is the difference between bytes and unicode strings and why must socket data be decoded?",
        "target_pages": ["149", "150", "155"],
        "keywords": ["encode", "decode", "UTF-8", "bytes", "unicode"],
    },

    # Category 8: JSON, XML, and data formats
    {
        "id": 32,
        "category": "JSON, XML & web services",
        "query": "How is XML structured with elements, tags, and attributes according to Chapter 13?",
        "target_pages": ["161", "162", "167"],
        "keywords": ["XML", "tags", "attributes", "ElementTree"],
    },
    {
        "id": 33,
        "category": "JSON, XML & web services",
        "query": "How do you parse XML using xml.etree.ElementTree and find text within nodes?",
        "target_pages": ["162", "163", "168"],
        "keywords": ["fromstring", "find", "get", "ElementTree"],
    },
    {
        "id": 34,
        "category": "JSON, XML & web services",
        "query": "Why is JSON often preferred over XML for modern web services and APIs?",
        "target_pages": ["165", "166", "171"],
        "keywords": ["JSON", "JavaScript Object Notation", "simpler", "native mapping to dictionaries"],
    },
    {
        "id": 35,
        "category": "JSON, XML & web services",
        "query": "How do you parse a JSON string into Python dictionaries and lists using json.loads?",
        "target_pages": ["166", "167", "172"],
        "keywords": ["json.loads", "json", "dictionary", "list"],
    },

    # Category 9: Database and SQL (SQLite)
    {
        "id": 36,
        "category": "Database & SQL",
        "query": "What is SQLite and how do you connect to a database file and create a cursor in Python?",
        "target_pages": ["183", "184", "189"],
        "keywords": ["sqlite3", "connect", "cursor", "database"],
    },
    {
        "id": 37,
        "category": "Database & SQL",
        "query": "What is the purpose of primary keys and foreign keys in relational database modeling?",
        "target_pages": ["195", "196", "201"],
        "keywords": ["primary key", "foreign key", "logical key", "relational database"],
    },
    {
        "id": 38,
        "category": "Database & SQL",
        "query": "How does the SQL JOIN clause link data across multiple tables using foreign keys?",
        "target_pages": ["199", "200", "205"],
        "keywords": ["JOIN", "ON", "multiple tables", "foreign key"],
    },

    # Category 10: Figures, diagrams, and structure
    {
        "id": 39,
        "category": "Figures & structure",
        "query": "What does Figure 1.1 or 1.3 in Chapter 1 show about computer hardware architecture?",
        "target_pages": ["3", "4", "5", "9", "10", "15"],
        "keywords": ["CPU", "Central Processing Unit", "Main Memory", "Secondary Memory", "Input and Output Devices"],
    },
    {
        "id": 40,
        "category": "Figures & structure",
        "query": "How is a relational database schema organized according to the data modeling diagrams in Chapter 15?",
        "target_pages": ["194", "195", "196", "203", "204", "205", "206", "209", "210", "211", "212"],
        "keywords": ["schema", "User table", "Course table", "Member table", "many-to-many"],
    },
]
