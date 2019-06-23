# =========================================================================== #
# user variables setup.
# feel free to adjust to your requirements

# --------------------------------------------------------------------------- #
# Compiler setup

CXX      = nvcc
CXXFLAGS = -std=c++14 -rdc=true -I $(LIBDIR)
LDFLAGS  = -lm -lcurand

LIBDIR = lib
SRCDIR = src
INCDIR = inc
OBJDIR = obj
EXEDIR = .

# --------------------------------------------------------------------------- #
# Project Data setup

EXTENSION_CODE   = .cu
EXTENSION_HEADER = .h

EXENAME = Galaxy

# --------------------------------------------------------------------------- #
# GIT setup

GIT_BASE = https://github.com
GIT_USERNAME = TheBlueChameleon
GIT_REPOSITORY = Galaxy
GIT_ADDITIONAL = "./makefile"
	# ./tex/*.tex ./makefile
	# ... list files here that you want to be part of the repository that are not already part of the compilable stuff
	# separate them by whitespaces
	# toggle to comment if you don't have any additional files to be added.
# GIT_EXCLUDE = 
	# ... list files here that you do not want to be part of the repository
	# separate by whitespaces
	# toggle to comment if you don't want to exclude any files

# =========================================================================== #
# Path Setup. Be sure that you really understand what you're doint if you edit anything below this line

DIRECTORIES = $(subst $(SRCDIR),$(OBJDIR),$(shell find $(SRCDIR) -type d))
	# pathes for files to be included into the compile/link procedure.
	# subst: "substitute PARAMETER_1 by PARAMETER_2 in PARAMETER_3.
	# shell find -type d lists only directories. find works recursively.
	# => load from SRCDIR and OBJDIR with all subdirectories

SRC     = $(wildcard $(SRCDIR)/*$(EXTENSION_CODE)) $(wildcard $(SRCDIR)/**/*$(EXTENSION_CODE))
	# list of all files in src, including subdirectories
INC     = $(wildcard $(INCDIR)/*$(EXTENSION_HEADER)) $(wildcard $(INCDIR)/**/*$(EXTENSION_HEADER))
	# same for includes
OBJ     = $(SRC:$(SRCDIR)/%$(EXTENSION_CODE)=$(OBJDIR)/%.o)
	# defines analogy relation?

# --------------------------------------------------------------------------- #
# Colour constants

COLOR_END	= \033[m

COLOR_RED	= \033[0;31m
COLOR_GREEN	= \033[0;32m
COLOR_YELLOW	= \033[0;33m
COLOR_BLUE	= \033[0;34m
COLOR_PURPLE	= \033[0;35m
COLOR_CYAN	= \033[0;36m
COLOR_GREY	= \033[0;37m

COLOR_LRED	= \033[1;31m
COLOR_LGREEN	= \033[1;32m
COLOR_LYELLOW	= \033[1;33m
COLOR_LBLUE	= \033[1;34m
COLOR_LPURPLE	= \033[1;35m
COLOR_LCYAN	= \033[1;36m
COLOR_LGREY	= \033[1;37m

MSG_OK		= $(COLOR_LGREEN)[SUCCES]$(COLOR_END)
MSG_WARNING	= $(COLOR_LYELLOW)[WARNING]$(COLOR_END)
MSG_ERROR	= $(COLOR_LRED)[ERROR]$(COLOR_END)

# =========================================================================== #
# procs

define fatboxtop
	@printf "$(COLOR_BLUE)"
	@printf "#=============================================================================#\n"
	@printf "$(COLOR_END)"
endef
# ........................................................................... #
define fatboxbottom
	@printf "$(COLOR_BLUE)"
	@printf "#=============================================================================#\n"
	@printf "$(COLOR_END)"
endef
# ........................................................................... #
define fatboxtext
	@printf "$(COLOR_BLUE)"
	@printf "# "
	@printf "$(COLOR_LGREY)"
	@printf "%-75b %s" $(1)
	@printf "$(COLOR_BLUE)"
	@printf "#\n"
	@printf "$(COLOR_END)"
	
endef
# --------------------------------------------------------------------------- #
define boxtop
	@printf "$(COLOR_BLUE)"
	@printf "+-----------------------------------------------------------------------------+\n"
	@printf "$(COLOR_END)"
endef
# ........................................................................... #
define boxbottom
	@printf "$(COLOR_BLUE)"
	@printf "+-----------------------------------------------------------------------------+\n"
	@printf "$(COLOR_END)"
endef
# ........................................................................... #
define boxtext
	@printf "$(COLOR_BLUE)"
	@printf "| "
	@printf "$(COLOR_LGREY)"
	@printf "%-75b %s" $(1)
	@printf "$(COLOR_BLUE)"
	@printf "|\n"
	@printf "$(COLOR_END)"
endef
# --------------------------------------------------------------------------- #
define fatbox
	$(call fatboxtop)
	$(call fatboxtext, $(1))
	$(call fatboxbottom)
endef
# ........................................................................... #
define box
	$(call boxtop)
	$(call boxtext, $(1))
	$(call boxbottom)
endef

# =========================================================================== #
# targets

.PHONY: intro all 

# --------------------------------------------------------------------------- #
# compound targets

all:   intro generate extro
new:   clean intro generate extro
run:   intro generate extro execute
grind: intro generate extro valgrind
gitstart: gitinit gitadds gitsetmasterhint

# --------------------------------------------------------------------------- #
# visual feedback

intro:
	@clear
	$(call fatbox, "attempting to make")
	@printf "$(COLOR_GREY)  "
	@date
	@echo ""
	
# ........................................................................... #
extro:
	$(call fatbox, "make done")
	@printf "$(COLOR_GREY)  "
	@date
	@echo ""
	
	
# --------------------------------------------------------------------------- #
# create executable file

generate: $(EXENAME)
# ........................................................................... #
$(OBJDIR)/%.o: $(SRCDIR)/%$(EXTENSION_CODE)                         # compile #
	$(call boxtop)
	$(call boxtext, "attempting to compile...")
	
	@mkdir -p $(DIRECTORIES)
	
	@printf "$(COLOR_BLUE)"
	@printf "| "
	@printf "$(COLOR_LBLUE)"
	@printf "%-75b %s" "  Compiling:  $(COLOR_LYELLOW)$<$(COLOR_END)"
	
	@-$(CXX) $(CXXFLAGS) -c $< -o $@ -I $(INCDIR)

	@printf "%-20b" "$(MSG_OK)"
	@printf "$(COLOR_BLUE)|\n"
	
	$(call boxtext, "done.")
	$(call boxbottom)
	
# ........................................................................... #
$(EXENAME): $(OBJ)                                                     # link #
	$(call boxtop)
	$(call boxtext, "attempting to link...")
	
	@mkdir -p $(EXEDIR)
	
	@printf "$(COLOR_BLUE)"
	@printf "| "
	@printf "$(COLOR_LBLUE)"
	@printf "%-85b %s" "  Linking:  $(COLOR_LYELLOW)$<$(COLOR_END)"
	@printf "$(COLOR_BLUE)|\n"
	
	@$(CXX) $^ -o $(EXEDIR)/$(EXENAME) $(LDFLAGS)
	
	$(call boxtext, "done.")
	$(call boxtop)
	
	
	@printf "$(COLOR_BLUE)"
	@printf "| "
	@printf "$(COLOR_LBLUE)"
	@printf "%-81b %s " "Executable: $(COLOR_LYELLOW)$(EXEDIR)/$(EXENAME)"
	@printf "$(COLOR_BLUE)|\n"
	
	$(call boxbottom)
	
# --------------------------------------------------------------------------- #
# run variations

execute:
	@./$(EXEDIR)/$(EXENAME)
	
# ........................................................................... #
valgrind :
	@valgrind ./$(EXEDIR)/$(EXENAME)
	
# --------------------------------------------------------------------------- #
# delete the object directory

clean:
	@printf "$(COLOR_LCYAN)"
	@echo "#=============================================================================#"
	@echo "# attempting to clean...                                                      #"
	
	@rm -rf $(OBJDIR)
	@rm -f $(EXEDIR)/$(EXENAME)
	
	@echo "# done.                                                                       #"
	@echo "#=============================================================================#"
	@echo ""
	
# --------------------------------------------------------------------------- #
# git versioning support

gitinit :
	@clear
	$(call fatbox, "Setting up GIT repository with name \"$(GIT_REPOSITORY)\"")
	@echo ""
	@echo "Note that this assumes there exists a repository named"
	@echo "   $(COLOR_CYAN)$(GIT_REPOSITORY)$(COLOR_END)"
	@echo "under the adress"
	@echo "   $(COLOR_CYAN)$(GIT_BASE)/$(GIT_USERNAME)$(COLOR_END)"
	@echo ""
	
	@git init
	@git remote add origin $(GIT_BASE)/$(GIT_USERNAME)/$(GIT_REPOSITORY).git
	
# ........................................................................... #
gitadds :
	@git add ./$(SRCDIR)
	@git add ./$(INCDIR)
	
	@git config credential.helper store
	@git config --global credential.helper 'cache --timeout 7200'
	
	@if [ -n "$(GIT_ADDITIONAL)" ]; then git add $(GIT_ADDITIONAL); fi           # -n: true if string length is nonzero 
	@if [ -n "$(GIT_EXCLUDE)" ]; then git rm -r --cached $(GIT_EXCLUDE); fi
	
	@echo ""
	@git status
	
	@echo "Reader for checkin. Type"
	@echo "   $(COLOR_CYAN)git commit -m \"version comments\"$(COLOR_END)"
	@echo "   $(COLOR_CYAN)git push$(COLOR_END)"
	@echo "to upload the current status"
	@echo ""
	
# ........................................................................... #
gitremove :
	@printf "$(COLOR_LCYAN)"
	@echo "#=============================================================================#"
	@echo "# Removing all local data for a GIT repository                                #"
	@echo "# This will not affect the data in the repository.                            #"
	
	@rm -rf ./.git
	
	@echo "# done.                                                                       #"
	@echo "#=============================================================================#"
	@echo ""
# ........................................................................... #
gitsetmasterhint :
	@echo "type"
	@echo "   $(COLOR_CYAN)make gitsetmaster$(COLOR_END)"
	@echo "to finalize the process after adjusting the staging area."
	@echo ""
# ........................................................................... #
gitsetmaster :
	@git commit -m "initial commit"
	@git push --set-upstream origin master
# --------------------------------------------------------------------------- #
# help section

vars :
	@clear
	$(call fatbox, "variables dump:")
	
	@echo "source code extension    : $(EXTENSION_CODE)"
	@echo "header files extension   : $(EXTENSION_HEADER)"
	@echo ""
	
	@echo "executable file name     : $(EXENAME)"
	@echo ""
	@echo "source code  directory   : $(SRCDIR)"
	@echo "include file directory   : $(INCDIR)"
	@echo "object file  directory   : $(OBJDIR)"
	@echo "binary       directory   : $(EXEDIR)"
	@echo ""
	
	@echo "binary source directories: $(DIRECTORIES)"
	@echo "source code files        :"
	@echo "   $(SRC)"
	@echo "header files             :"
	@echo "   $(INC)"
	@echo "object code files        :"
	@echo "   $(OBJ)"
	@echo ""
	
	@echo "GIT host      : $(GIT_BASE)"
	@echo "GIT username  : $(GIT_USERNAME)"
	@echo "GIT repository: $(GIT_REPOSITORY)"
	@echo "additional files to be added:"
	@echo "   $(GIT_ADDITIONAL)"
	@echo "files to be excluded        :"
	@echo "   $(GIT_EXCLUDE)"
	
	$(call fatbox, "done.")
	
# ........................................................................... #
help :
	@echo "This makefile supports the following targets:"
	@echo "$(COLOR_YELLOW)Compile targets$(COLOR_END)"
	@echo "* $(COLOR_LCYAN)clean$(COLOR_END)"
	@echo "   removes $(OBJDIR) and its contents."
	@echo "* $(COLOR_LCYAN)all$(COLOR_END)"
	@echo "   compiles any files that may have changed since the last time this was run"
	@echo "* $(COLOR_LCYAN)new$(COLOR_END)"
	@echo "   compiles all files, regardless of whether they may have changed since the last time this was run"
	@echo "* $(COLOR_LCYAN)run$(COLOR_END)"
	@echo "   compiles all files and runs the program thereafter"
	@echo "* $(COLOR_LCYAN)grind$(COLOR_END)"
	@echo "   compiles all files and runs the program via valgrind"
	@echo ""
	@echo "$(COLOR_YELLOW)GIT targets$(COLOR_END)"
	@echo "* $(COLOR_LCYAN)gitstart$(COLOR_END)"
	@echo "   sets up the working directory for git and prepares an initial commit"
	@echo "* $(COLOR_LCYAN)gitsetmaster$(COLOR_END)"
	@echo "   sets the master branch and pushes the initial commit. $(COLOR_LRED)Run only once$(COLOR_END)"
	@echo "* $(COLOR_LCYAN)gitadds$(COLOR_END)"
	@echo "   adds files to the staging area."
	@echo "* $(COLOR_LCYAN)gitremove$(COLOR_END)"
	@echo "   removes all local data of the versioning system"
	@echo ""
	@echo "$(COLOR_YELLOW)Info targets$(COLOR_END)"
	@echo "* $(COLOR_LCYAN)vars$(COLOR_END)"
	@echo "   show variables generated and set by this script"
	@echo "* $(COLOR_LCYAN)help$(COLOR_END)"
	@echo "   show this help"
	@echo ""
	
	@echo "Note that you can create compound targets such as:"
	@echo "   $(COLOR_LCYAN)make clean run$(COLOR_END)"
	
