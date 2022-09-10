PKG := $(shell cat DESCRIPTION | awk '$$1 == "Package:" { print $$2 }')
VER := $(shell cat DESCRIPTION | awk '$$1 == "Version:" { print $$2 }')

SRC := $(wildcard src/*.cc)
HDR := $(wildcard src/*.hh)

all: $(PKG)_$(VER).tar.gz

clean:
	rm -f src/*.o src/*.so
	rm -f $(PKG)_$(VER).tar.gz

$(PKG)_$(VER).tar.gz: $(SRC) $(HDR) .Rbuildignore
	rm -f src/*.so $@
	R -e "Rcpp::compileAttributes(verbose=TRUE)"
	R -e "usethis::use_roxygen_md(); roxygen2md::roxygen2md(); devtools::document()"
	R CMD build .

check: $(PKG)_$(VER).tar.gz
	R CMD check $<

install: $(PKG)_$(VER).tar.gz
	R CMD INSTALL $<

site:
	R -e "pkgdown::build_site()"

