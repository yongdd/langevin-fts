/*-----------------------------------------------------------------
! This is a parser implemented using regular expression (RE)
! and deterministic finite automata (DFA). This module reads input
! parameters from an input file. Each parameter pair is stored in
! vector, and retrieve it when 'get' is invoked.
!--------------------------------------------------------------------*/

#include "ParamParser.h"


//----------------- Constructor -----------------------------
ParamParser::ParamParser()
{
    bool finished = false;  // parsing is finished
}
//----------------- Destructor ------------------------------
ParamParser::~ParamParser()
{
    mtx.lock(); //lock threads
    // show parameters that are never used.
    std::cout<< "---------- Parameters that are never used ----------" << std::endl;
    for(unsigned int i=0; i<input_param_list.size(); i++)
    {
        if(param_use_count[i] == 0)
            std::cout<< input_param_list[i].var_name << std::endl;
    }
    mtx.unlock(); //unlock threads
}
//----------------- read_param_file -------------------------
void ParamParser::read_param_file(std::string param_file_name)
{
    std::string buf;
    int n_line;
    ParamPair input_param;

    mtx.lock(); //lock threads

    if(finished)
        return;

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
    

    std::cout<< "---------- Input Parameters ----------" << std::endl;
    for(unsigned i=0; i<input_param_list.size(); i++ )
    {
        std::cout<< input_param_list[i].var_name << " :";
        for(unsigned j = 0; j<input_param_list[i].values.size(); j++)
            std::cout<< " " << input_param_list[i].values[j];
        std::cout<< std::endl;
    }
    finished = true;
    mtx.unlock(); //unlock threads
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
    for(unsigned i=0; i<= buf.length(); i++)
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
            std::cout<< "  Syntax Error at: "<< n_line << " " << i+1 << std::endl;
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
    else if (('0' <= ch && ch <= '9') || ch == '-' ) // digit
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
    for(unsigned int i = 0; i<input_param_list.size(); i++)
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
int ParamParser::search_param_idx(std::string str_name, unsigned int idx)
{
    if(!finished){
        std::cout << "Call ParamParser::read_param_file first." << std::endl;
	exit(-1);
    }

    for(unsigned int i = 0; i<input_param_list.size(); i++)
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
    if( loc >= 0)
    {
        std::stringstream ss(input_param_list[loc].values[idx]);
        //std::cout << "get " << input_param_list[loc].values[idx] << std::endl ;
        ss >> param_value;
        //std::cout << "get " << param_value  << std::endl ;
        return true;
    }
    else
        return false;
}
bool ParamParser::get(std::string param_name, int *param_value, int length)
{
    int loc;
    loc = search_param_idx(param_name, length-1);
    if( loc >= 0)
    {
        for(int i=0; i<length; i++)
        {
            std::stringstream ss(input_param_list[loc].values[i]);
            ss >> param_value[i];
        }
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
    if( loc >= 0)
    {
        std::stringstream ss(input_param_list[loc].values[idx]);
        ss >> param_value;
        return true;
    }
    else
        return false;
}
bool ParamParser::get(std::string param_name, double *param_value, int length)
{
    int loc;
    loc = search_param_idx(param_name, length-1);
    if( loc >= 0)
    {
        for(int i=0; i<length; i++)
        {
            std::stringstream ss(input_param_list[loc].values[i]);
            ss >> param_value[i];
        }
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
    if( loc >= 0)
    {
        std::stringstream ss(input_param_list[loc].values[idx]);
        ss >> param_value;
        return true;
    }
    else
        return false;
}
