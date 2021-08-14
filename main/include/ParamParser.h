/*-----------------------------------------------------------------
! This is a parser implemented using regular expression (RE)
! and deterministic finite automata (DFA). This module reads input
! parameters from an input file. Each parameter pair is stored in
! vector, and retrieve it when 'get' is invoked.
!-------------- Regular Expression --------------------------------
! s = space
! t = tab
!
! word,   w = [a-zA-Z_]
! digit,  i = [0-9]
! assign, g = [=:]
! dot,        .
! quote,      "
! exponent, x = (d|D|e|E)
! sign, p = (+|-)
!
! blank,  b = (s|t)
! name,   c = (w|i)+(.(w|i)+)?
! number, n = p?i+(.i*)?(xp?i+)?
! string, s = "(w|i|.)*"
! value,  v = (n|s)
! all,    a = [all chatracter]
!
! Syntax :
! (b*cb*gb*v(b+v)*)?b*(#a*)?
!
!
!-------------- Transition DFA Table---------------------------------
!            |       | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
! state\input| Type  | b |w^x| g | p | i | . | x | " | # | Other|
!-------------------------------------------------------
!   1        | accept| 1 |12 |   |   |12 |   |12 |   |11 |    |
!   2(V)     |       |   | 2 |   |   | 2 | 2 | 2 | 3 |   |    |
!   3(V)     | accept| 7 |   |   |   |   |   |   |   |11 |    |
!   4(V)     | accept| 7 |   |   |   | 4 |   |10 |   |11 |    |
!   5(V)     | accept| 7 |   |   |   | 5 |   |   |   |11 |    |
!   6(V)     |       |   |   |   |   | 5 |   |   |   |   |    |
!   7        | accept| 7 |   |   | 9 | 8 |   |   | 2 |11 |    |
!   8(V)     | accept| 7 |   |   |   | 8 | 4 |10 |   |11 |    |
!   9(V)     |       |   |   |   |   | 8 |   |   |   |   |    |
!   10(V)    |       |   |   |   | 6 | 5 |   |   |   |   |    |
!   11       | accept|   |   |   |   |   |   |   |   |   |    |
!   12(S)    |       |15 |12 |14 |   |12 |13 |12 |   |   |    |
!   13(S)    |       |   |16 |   |   |16 |   |16 |   |   |    |
!   14       |       |14 |   |   | 9 | 8 |   |   | 2 |   |    |
!   15       |       |15 |   |14 |   |   |   |   |   |   |    |
!   16(S)    |       |15 |16 |14 |   |16 |   |16 |   |   |    |
!--------------------------------------------------------------------
! Blank cells in above table are all Syntax Error
!--------------------------------------------------------------------*/



#ifndef PARAM_PARSER_H_
#define PARAM_PARSER_H_

#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <fstream>
#include <sstream>
#include <mutex>

// Design Pattern : Singleton (Scott Meyer)

class ParamParser
{

private :

    // variable and value.
    struct ParamPair
    {
        std::string var_name;
        std::vector<std::string> values;
    };

    enum CharType
    {
        TYPE_BLANK,
        TYPE_WORD, // alplhabat except 'eEdD'
        TYPE_ASSIGN,
        TYPE_SIGN,
        TYPE_DIGIT,
        TYPE_DOT,
        TYPE_EXPONENT, // 'eEdD'
        TYPE_QUOTE,
        TYPE_END,
        TYPE_OTHER,
    };

    // Transition Table of DFA
    std::array<std::array<int, 10>,16> dfa_transit = {{
            { 1,12, 0, 0,12, 0,12, 0,11, 0},
            { 0, 2, 0, 0, 2, 2, 2, 3, 0, 0},
            { 7, 0, 0, 0, 0, 0, 0, 0,11, 0},
            { 7, 0, 0, 0, 4, 0,10, 0,11, 0},
            { 7, 0, 0, 0, 5, 0, 0, 0,11, 0},
            { 0, 0, 0, 0, 5, 0, 0, 0, 0, 0},
            { 7, 0, 0, 9, 8, 0, 0, 2,11, 0},
            { 7, 0, 0, 0, 8, 4,10, 0,11, 0},
            { 0, 0, 0, 0, 8, 0, 0, 0, 0, 0},
            { 0, 0, 0, 6, 5, 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {15,12,14, 0,12,13,12, 0, 0, 0},
            { 0,16, 0, 0,16, 0,16, 0, 0, 0},
            {14, 0, 0, 9, 8, 0, 0, 2, 0, 0},
            {15, 0,14, 0, 0, 0, 0, 0, 0, 0},
            {15,16,14, 0,16, 0,16, 0, 0, 0}
        }
    };

    // Whether to save a character in each state
    std::array<int, 16> state_store_char = {0,1,1,1,1,1,0,1,1,1,0,1,1,0,0,1};

    // parameters will be stored here.
    std::vector<ParamPair> input_param_list;
    // for debuging, this records counts how many time each parameter are called.
    std::vector<int> param_use_count;

    bool finished;  // parsing is finished
    std::mutex mtx; // mutex for thread safe.
                    // this is neccesary only if you call
		    // ream_param_file using multi threads

    bool line_has_parsed(std::string buf, ParamPair &input_param, int n_line);
    void insert_param(ParamPair input_param);
    int search_param_idx(std::string param_name, unsigned int idx);
    int get_char_type(char ch);

    ParamParser();
    ~ParamParser();
    // Disable copy constructor
    ParamParser(const ParamParser &) = delete;
    ParamParser& operator= (const ParamParser &) = delete;

public:

    static ParamParser& get_instance() {
        static ParamParser* instance = new ParamParser();
	return *instance;
    };

    void read_param_file(std::string param_file_name); 

    bool get(std::string param_name, int &param_value, int idx=0);
    bool get(std::string param_name, int *param_value, int length);
    bool get(std::string param_name, double &param_value, int idx=0);
    bool get(std::string param_name, double *param_value, int length);
    bool get(std::string param_name, std::string &param_value, int idx=0);

};

#endif
