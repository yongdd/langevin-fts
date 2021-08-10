/*--------------------------------------------------------------------
! This is a parser implemented using regular expression (RE)
! and deterministic finite automata (DFA). This module reads input
! parameters from an input file as well as command line arguments.
! Each parameter pair are stored in static arrays, and retrieve them
! when 'pp_get' is invoked.
!-------------- Regular Expression ---------------------------------
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
!------------------------------------------------------------------------
! Blank cells in above table are all Syntax Error
!-----------------------------------------------------------------------*/

#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <fstream>
#include <sstream>

class ParamParser
{

private :

    // variable and value.
    struct ParamPair
    {
        std::string var_name;
        std::vector<std::string> values;
    };

    enum Word
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

    const int MAX_STRING_LENGTH{256};
    const int MAX_TOKEN_LENGTH{64};
    const int MAX_TOKEN_NUM{256};
    const int MAX_ITEM{256};

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

    bool finished = false;  // parsing is finished

    bool line_has_parsed(std::string buf, ParamPair &input_param, int n_line);
    void insert_param(ParamPair input_param);
    int search_param_idx(std::string param_name, int idx);
    int get_char_type(char ch);

public:

    ParamParser(std::string param_file_name);
    ~ParamParser();
    
    bool get(std::string param_name, int &param_value, int idx=0);
    bool get(std::string param_name, double &param_value, int idx=0);
    bool get(std::string param_name, std::string &param_value, int idx=0);

};

//----------------- Constructor -----------------------------
ParamParser::ParamParser(std::string param_file_name)
{
    std::string buf;
    int i, j, file_stat;
    int n_line, arg_num;
    ParamPair input_param;
    
    input_param_list.clear();
    param_use_count.clear();

    // Read parameters from input files
    std::ifstream param_file(param_file_name);
    if (!param_file.is_open())
    {
        std::cerr << "Could not open the file - '"
                  << param_file_name << "'" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    n_line = 1;
    while (std::getline(param_file, buf))
    {
        //std::cout<< buf << std::endl;
        if( line_has_parsed(buf, input_param, n_line))
            insert_param(input_param);
        param_use_count.push_back(0);
        n_line++; // line number
    }
    param_file.close();

    std::cout<< "--- Input Parameters ---" << std::endl;
    for(i=0; i<input_param_list.size(); i++ )
    {
        std::cout<< input_param_list[i].var_name << " :";
        for(j = 0; j<input_param_list[i].values.size(); j++)
            std::cout<< " " << input_param_list[i].values[j];
        std::cout<< std::endl;
    }
    std::cout<< "-----------------------" << std::endl;
}
//---------------- line_has_parsed ------------------------
bool ParamParser::line_has_parsed(std::string buf, ParamPair &input_param, int n_line)
{
    std::vector<std::string> tokens;
    int i, tt, cur_state, old_state;
    int n_values;

    //tokens(:) = ""
    input_param.var_name = "";

    cur_state = 1;
    n_values = 0;
    tokens.push_back("");
    for(i=0; i<= buf.length(); i++)
    {
        if ( i < buf.length())
            tt = get_char_type(buf[i]);
        else
            tt = TYPE_END;

        old_state = cur_state;
        cur_state = dfa_transit[old_state-1][tt];
        //std::cout<< "tt, old, cur : " << tt << " " << old_state << " " << cur_state << std::endl;

        // Syntax Errror
        if( cur_state == 0)
        {
            std::cout<< "  Syntax Error at: "<< n_line << " " << i << std::endl;
            return false;
        }
        // Making token
        else if( state_store_char[cur_state-1] == 1 )
        {
            tokens[n_values] += buf[i];
            //std::cout<< "buf[i] " << n_values << " " << tokens[n_values] << std::endl;
        }
        // Token is made
        else if( state_store_char[cur_state-1] == 0
                 && state_store_char[old_state-1] == 1)
        {
            tokens.push_back("");
            n_values++;
        }
        if (tt == TYPE_END)
            break;

    }

    //for(i=0; i < tokens.size()-1; i++)
    //    std::cout<< "tokens "<< i << " " << tokens[i] << std::endl;

    if (n_values > 0)
    {
        input_param.var_name = tokens[0];
        input_param.values.clear();
        for(i=1; i<n_values; i++)
            input_param.values.push_back(tokens[i]);
    }
    if (input_param.var_name == "")
        return false;

    return  true;
}
/*
!---------------- pp_get_from_string_int_alloc ------------------------
  ! integer allocatable
  logical function pp_get_from_string_int_alloc(buf, var, n_line)
    character(len=*), intent(in) :: buf
    integer, intent(inout), allocatable :: var(:)
    integer, intent(in) :: n_line
    type(param_pair) :: input_param

    pp_get_from_string_int_alloc = .False.
    if( line_has_parsed(buf, input_param, n_line)) then
      allocate(var(size(input_param%values)))
      read (input_param%values, *) var
      pp_get_from_string_int_alloc = .True.
    end if
    if(allocated(input_param%values)) deallocate(input_param%values)
  end function

  ! real
  logical function pp_get_from_string_real(buf, var, idx, n_line)
    character(len=*), intent(in) :: buf
    real(kind=8), intent(inout) :: var
    integer, intent(in) :: idx, n_line
    type(param_pair) :: input_param

    pp_get_from_string_real = .False.
    if( line_has_parsed(buf, input_param, n_line)) then
      if(size(input_param%values) >= idx) then
        read (input_param%values(idx), *) var
        pp_get_from_string_real = .True.
      end if
    end if
  end function
*/
//---------------- get_char_type ------------------------
// character comparison is conducted based on ASCII code
int ParamParser::get_char_type(char ch)
{
    if(ch == ' ' || ch == '	') // space and Tab
        return TYPE_BLANK;
    else if (ch == '-' || ch == '+') // plus minus sign
        return TYPE_SIGN;
    else if (ch == 'e' || ch == 'E' || // exponent
             ch == 'd' || ch == 'D')
        return TYPE_EXPONENT;
    else if (ch == '_' || // alpha, underscore except 'e','E','c','D'
             ('A' <= ch && ch <= 'Z') ||
             ('a' <= ch && ch <= 'z'))
        return TYPE_WORD;
    else if ('0' <= ch && ch <= '9' || ch == '-' ) // digit
        return TYPE_DIGIT;
    else if (ch == '.' ) // dot
        return TYPE_DOT;
    else if ( ch == '"') // double quote
        return TYPE_QUOTE;
    else if ( ch =='=' ||  ch ==':') // assign
        return TYPE_ASSIGN;
    else if ( ch =='#') // comment #
        return TYPE_END;
    else // invalid character
        return TYPE_OTHER;
}
//----------------- insert_param --------------------------
void ParamParser::insert_param(ParamPair new_param)
{
    for(int i = 0; i<input_param_list.size(); i++)
    {
        if( new_param.var_name == input_param_list[i].var_name)
        {
            std::cout<< "Warning! '" << new_param.var_name
                     << "' is overwritten due to duplicated input parameter." << std::endl;
            input_param_list[i].values = new_param.values;
            return;
        }
    }
    input_param_list.push_back(new_param);
}
//----------------- search_param_idx ----------------------------
int ParamParser::search_param_idx(std::string str_name, int idx)
{
    for(int i = 0; i<input_param_list.size(); i++)
    {
        if( str_name == input_param_list[i].var_name)
        {
            if ( idx < input_param_list[i].values.size() )
            {
                // Update count. To prevent overflow min is used.
                param_use_count[i] = std::min(param_use_count[i] + 1, 100000);
                return i;
            }
            else
                return -1;
        }
    }
    return -1;
}
//---------------------- get_int -------------------------------
bool ParamParser::get(std::string param_name, int& param_value, int idx)
{
    int loc;
    loc = search_param_idx(param_name, idx);
    //std::cout << "loc " << loc << std::endl ;
    if( loc >= 0) {
      std::stringstream ss(input_param_list[loc].values[idx]);
      //std::cout << "get " << input_param_list[loc].values[idx] << std::endl ;
      ss >> param_value;
      //std::cout << "get " << param_value  << std::endl ;
      return true;
    }
    else
      return false;
}
//---------------------- get_double -------------------------------
bool ParamParser::get(std::string param_name, double& param_value, int idx)
{
    int loc;
    loc = search_param_idx(param_name, idx);
    if( loc >= 0) {
      std::stringstream ss(input_param_list[loc].values[idx]);
      ss >> param_value;
      return true;
    }
    else
      return false;
}
//---------------------- get_string -------------------------------
bool ParamParser::get(std::string param_name, std::string& param_value, int idx)
{
    int loc;
    loc = search_param_idx(param_name, idx);
    if( loc >= 0) {
      std::stringstream ss(input_param_list[loc].values[idx]);
      ss >> param_value;
      return true;
    }
    else
      return false;
}
//---------------------- Destructor -------------------------------
ParamParser::~ParamParser()
{
    // show parameters that are never used.
    std::cout<< "------ Parameters that are never used ------" << std::endl;
    for(int i=0; i<input_param_list.size(); i++){
      if(param_use_count[i] == 0)
        std::cout<< input_param_list[i].var_name << std::endl;
    }
}

int main()
{
    int i;
    double d;
    std::string s;
    ParamParser param_parser("inputs");
    param_parser.get("geometry.grids", i);
    std::cout<< "geometry.grids[0]: "<< i << std::endl;
    
    param_parser.get("chain.chiN", d);
    std::cout<< "chain.chiN: "<< d << std::endl;
    
    param_parser.get("filename", s);
    std::cout<< "filename: "<< s << std::endl;
    
    return 0;
}
