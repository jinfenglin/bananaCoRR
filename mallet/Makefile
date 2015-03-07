MALLET_DIR = $(shell pwd)

JAVAC = javac
JAVA_FLAGS = \
-classpath "$(MALLET_DIR)/class:$(MALLET_DIR)/lib/mallet-deps.jar" \
-sourcepath "$(MALLET_DIR)/src" \
-g:lines,vars,source \
-d $(MALLET_DIR)/class \
-J-Xmx200m -source 1.4

JAVADOC = javadoc
JAVADOC_FLAGS = -J-Xmx300m

all: class link-resources
	$(JAVAC) $(JAVA_FLAGS) `find src -name '*.java'`

javadoc: html class
	$(JAVADOC) $(JAVADOC_FLAGS) -classpath "$(MALLET_DIR)/class:$(MALLET_DIR)/lib/mallet-deps.jar" -d $(MALLET_DIR)/html -sourcepath $(MALLET_DIR)/src -source 1.4 -subpackages edu

mccallum: class link-resources
	$(JAVAC) $(JAVA_FLAGS) `find src \( \( \! -path 'src/edu/umass/cs/mallet/users/*' \) -o -path 'src/edu/umass/cs/mallet/users/mccallum/*' \) -name '*.java'`

mccallum2: class
	$(JAVAC) $(JAVA_FLAGS) `find src/edu/umass/cs/mallet/users/mccallum -name '*.java'` `find src/edu/umass/cs/mallet/base -name '*.java'` 

pinto: class link-resources
	$(JAVAC) $(JAVA_FLAGS) `find src \( \( \! -path 'src/edu/umass/cs/mallet/users/*' \) -o -path 'src/edu/umass/cs/mallet/users/pinto/*' \) -name '*.java'`

culotta: class link-resources
	$(JAVAC) $(JAVA_FLAGS) `find src \( \( \! -path 'src/edu/umass/cs/mallet/users/*' \) -o -path 'src/edu/umass/cs/mallet/users/culotta/*' \) -name '*.java'`

feng: class link-resources
	$(JAVAC) $(JAVA_FLAGS) `find src \( \( \! -path 'src/edu/umass/cs/mallet/users/*' \) -o -path 'src/edu/umass/cs/mallet/users/feng/*' \) -name '*.java'`

# I think the above can be abstracted somewhat.  Let me have a go at this...
mine: class link-resources
	$(JAVAC) $(JAVA_FLAGS) `find src \( \( \! -path 'src/edu/umass/cs/mallet/users/*' \) -o -path "src/edu/umass/cs/mallet/users/${USER}/*" \) -name '*.java'`

copy-resources: class
	cd src ; gtar --exclude CVS -cf - `find . -type d -name resources` | (cd ../class ; gtar -xf -)

# Soft link the resources directories in mallet/src into mallet/class
link-resources: class
	cd src ; for d in `find . -type d -name resources` ; do \
	  echo $$d ; \
	  mkdir -p `dirname ../class/$$d` ; \
	  rm -f ../class/$$d ; \
	  (cd ../class ; ln -s `echo $$d | sed 's,/[^/]*,/\.\.,g'`/src/$$d $$d ) ; \
	done

class:
	mkdir -p class

html:
	mkdir -p html

clean:
	rm -rf class/* lib/unpack

echo-classpath:
	export CLASSPATH=$(MALLET_DIR)/class

-include Makefile.local
