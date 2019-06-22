LIBDIR = lib
SRCDIR = src
INCDIR = inc
OBJDIR = obj
EXEDIR = .

CXX      = nvcc
CXXFLAGS = -std=c++14 -rdc=true -I $(LIBDIR)
LDFLAGS  = -lm -lcurand

DIRECTORIES = $(subst $(SRCDIR),$(OBJDIR),$(shell find $(SRCDIR) -type d))
	# pathes for files to be included into the compile/link procedure.
	# subst: "substitute PARAMETER_1 by PARAMETER_2 in PARAMETER_3.
	# shell find -type d lists only directories. find works recursively.
	# => load from SRCDIR and OBJDIR with all subdirectories

EXENAME = Galaxy
SRC     = $(wildcard $(SRCDIR)/*.cu) $(wildcard $(SRCDIR)/**/*.cu)
	# list of all files in src, including subdirectories
INC     = $(wildcard $(SRCDIR)/*.h) $(wildcard $(SRCDIR)/**/*.h)
	# same for includes
OBJ     = $(SRC:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)
	# defines analogy relation?

# =========================================================================== #

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

.PHONY: intro all 

# --------------------------------------------------------------------------- #
all:   intro generate extro
new:   clean intro generate extro
run:   intro generate extro execute
grind: intro generate extro valgrind
# --------------------------------------------------------------------------- #
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
generate: $(EXENAME)
# ........................................................................... #
# compile
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(call boxtop)
	$(call boxtext, "attempting to compile...")
	
	@mkdir -p $(DIRECTORIES)
	
	@printf "$(COLOR_BLUE)"
	@printf "| "
	@printf "$(COLOR_LBLUE)"
	@printf "%-75b %s" "  Compiling:  $(COLOR_LYELLOW)$<$(COLOR_END)"
	
	@printf $(CXX) $(CXXFLAGS) -c $< -o $@ -I $(INCDIR)
	@$(CXX) $(CXXFLAGS) -c $< -o $@ -I $(INCDIR)

	@printf "%-20b" "$(MSG_OK)"
	@printf "$(COLOR_BLUE)|\n"
	
	$(call boxtext, "done.")
	$(call boxbottom)
	
# ........................................................................... #
# link
$(EXENAME): $(OBJ)
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
execute:
	@./$(EXEDIR)/$(EXENAME)
	
# --------------------------------------------------------------------------- #
valgrind :
	@valgrind ./$(EXEDIR)/$(EXENAME)
	
# --------------------------------------------------------------------------- #
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
vars :
	@clear
	$(call fatbox, "variables dump:")
	
	@echo "SRCDIR: $(SRCDIR)"
	@echo "INCDIR: $(INCDIR)"
	@echo "OBJDIR: $(OBJDIR)"
	@echo "EXEDIR: $(EXEDIR)"

	@echo "DIRECTORIES: $(DIRECTORIES)"

	@echo "EXENAME: $(EXENAME)"
	@echo "SRC: $(SRC)"
	@echo "INC: $(INC)"
	@echo "OBJ: $(OBJ)"
	$(call fatbox, "done.")
