import sys
import ast
import numpy as np

# Read the parameter file
fp = open("scft_example.py", 'r')
code = fp.read()
# tree = ast.parse(code)
# print(ast.unparse(tree))

# Modify python code to read only parameters and to make input fields
lines = code.splitlines()
id = ""
new_lines = []
for line in lines:
    if "import scft" in line:
        continue
    elif "scft." in line or "lfts." in line:
        id = line.split()[0]
        continue
    elif (not id == "") and (id + ".") in line:
        continue
    new_lines.append(line)
read =  '\n'.join(new_lines)

# Execute modified code
exec(read)

# Print params
print(params["segment_lengths"])

# Print fields
for monomer_type in params["segment_lengths"]:
    print(f"w_{monomer_type}.shape: ", eval(f"w_{monomer_type}").shape)
    
# Generate PSCF 'param' file
nx = params["nx"]
pscf_param = """System{{
  Mixture{{
    nMonomer  {0}
    monomers[
""".format(len(params["segment_lengths"]))
monomer_to_idx = {}
for count, monomer_type in enumerate(params["segment_lengths"]):
    pscf_param += 6*" " + "{0}\n".format(params["segment_lengths"][monomer_type])
    monomer_to_idx[monomer_type] = count
pscf_param += 4*" " + "]\n"
pscf_param += 4*" " + "nPolymer  {0}\n".format(len(params["distinct_polymers"]))
for polymer in params["distinct_polymers"]:
    pscf_param += 4*" " + "Polymer{\n"
    if "v" in polymer["blocks"][0]:
        type = "branched"
    else:
        type = "linear"
    pscf_param += 6*" " + "type    " + type +"\n"
    pscf_param += 6*" " + "nBlock  {0}\n".format(len(polymer["blocks"]))
    pscf_param += 6*" " + "blocks\n"
    for block in polymer["blocks"]:
        if type == "linear":
            pscf_param   += 8*" " + "{0}  {1}\n".format(monomer_to_idx[block["type"]], block["length"])
        elif type == "branched":
            pscf_param   += 8*" " + "{0}  {1} {2} {3}\n".format(monomer_to_idx[block["type"]], block["length"], block["v"], block["u"])
    pscf_param += 6*" " + "phi  {0}\n".format(polymer["volume_fraction"])
    pscf_param += 4*" " + "}\n"
pscf_param += 4*" " + "ds  {0}".format(params["ds"])
pscf_param += """
  }
  Interaction{
    chi(
"""
for chi_n_pair in params["chi_n"]:
    monomer_pair = chi_n_pair.split(",")
    pscf_param += 6*" " + "{0}  {1}  {2}\n".format(monomer_to_idx[monomer_pair[0]], monomer_to_idx[monomer_pair[1]], params["chi_n"][chi_n_pair])
pscf_param += """    )
  }
  Domain{
    mesh    """
pscf_param += "    {0}".format("   ".join(map(str, nx)))
pscf_param += """
    lattice     orthorhombic
    groupName   P_1
  }}
  AmIteratorGrid{{
    epsilon  1.0e-5
    maxItr   1000
    maxHist  20
    isFlexible   {0}
  }}
}}\n""".format(1 if params["box_is_altering"] == True else 0)
# print(pscf_param)

# Generate PSCF 'in/omega' file
pscf_omega = """format   1   0
dim                 
                  {0}
crystal_system      
          orthorhombic
N_cell_param        
                  {1}
cell_param          
                  {2}
group_name          
    P_1
N_monomer           
                  {3}
mesh
                  {4}\n""".format(
    # "   ".join(map(str,monomer_to_idx.values())),
    len(nx),
    len(nx),
    "   ".join(map(str, params["lx"])),
    len(params["segment_lengths"]),
    "   ".join(map(str, nx)),
    )
S = len(params["segment_lengths"])
n_grid = np.prod(nx)
w = np.zeros([S, n_grid], dtype=np.float64)

for count, monomer_type in enumerate(params["segment_lengths"]):
    w[count,:] = np.reshape(np.reshape(eval(f"w_{monomer_type}"), nx).transpose(2,1,0), n_grid)

for i in range(w.shape[1]):
    for n in monomer_to_idx.values():
       pscf_omega += "  %15.10f" % (w[n,i])
    pscf_omega += "\n"
# print(pscf_omega)
       
# Write files
f = open("param", 'w')
f.write(pscf_param)
f.close()

f = open("omega", 'w')
f.write(pscf_omega)
f.close()