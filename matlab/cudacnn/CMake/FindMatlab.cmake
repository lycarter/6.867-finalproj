# Version slightly corrected to find latest Matlab versions. 
# Special thanks for Tom Lampert for fixes to make it work on OSX
# Mikhail Sirotenko (2012)
# - this module looks for Matlab
# Defines:
#  MATLAB_INCLUDE_DIR: include path for mex.h, engine.h
#  MATLAB_LIBRARIES:   required libraries: libmex, etc
#  MATLAB_MEX_LIBRARY: path to libmex.lib
#  MATLAB_MX_LIBRARY:  path to libmx.lib
#  MATLAB_ENG_LIBRARY: path to libeng.lib
#  MATLAB_MEXFILE_EXT: platform-specific extension for mex-files
#=============================================================================
# Copyright 2005-2009 Kitware, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# * Redistributions of source code must retain the above copyright
  # notice, this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright
  # notice, this list of conditions and the following disclaimer in the
  # documentation and/or other materials provided with the distribution.

# * Neither the names of Kitware, Inc., the Insight Software Consortium,
  # nor the names of their contributors may be used to endorse or promote
  # products derived from this software without specific prior written
  # permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#=============================================================================

SET(MATLAB_FOUND 0)

IF(NOT MATLAB_ROOT)
	SET(MATLAB_MEX_LIBRARY MATLAB_MEX_LIBRARY-NOTFOUND)
	SET(MATLAB_MX_LIBRARY MATLAB_MX_LIBRARY-NOTFOUND)
	SET(MATLAB_ENG_LIBRARY MATLAB_ENG_LIBRARY-NOTFOUND)
	IF(WIN32)
		#Find Matlab root. Suppose the Matlab version is between 7.1 and 8.15
		#SET(MATLAB_ROOT "")
		SET(MATLAB_INCLUDE_DIR "")
		SET(_MATLAB_INCLUDE_DIR "")		
		SET(REGISTRY_KEY_TMPL "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\TMPL_MAJOR.TMPL_MINOR;MATLABROOT]")
		FOREACH(MATLAB_MAJOR RANGE 7 8)
		FOREACH(MATLAB_MINOR RANGE 15)
			#STRING(REPLACE "TMPL_MINOR" "${MATLAB_MINOR}" REGISTRY_KEY_OUTP ${REGISTRY_KEY_TMPL})
			STRING(REPLACE "TMPL_MINOR" "${MATLAB_MINOR}" REGISTRY_KEY_OUTP ${REGISTRY_KEY_TMPL})
			STRING(REPLACE "TMPL_MAJOR" "${MATLAB_MAJOR}" REGISTRY_KEY_OUTP ${REGISTRY_KEY_OUTP})
			GET_FILENAME_COMPONENT(_MATLAB_ROOT ${REGISTRY_KEY_OUTP} ABSOLUTE)
			IF(NOT ${_MATLAB_ROOT} STREQUAL "/registry")
				SET(MATLAB_ROOT ${_MATLAB_ROOT})
			ENDIF()
		ENDFOREACH()
		ENDFOREACH()

	  IF(${CMAKE_GENERATOR} MATCHES "Visual Studio")
		IF(64BIT)
			SET(MATLAB_LIB_DIR "${MATLAB_ROOT}/extern/lib/win64/microsoft/")		
		ELSE(64BIT)
			SET(MATLAB_LIB_DIR "${MATLAB_ROOT}/extern/lib/win32/microsoft/")			
		ENDIF(64BIT)
	  ELSE(${CMAKE_GENERATOR} MATCHES "Visual Studio")
		IF(MATLAB_FIND_REQUIRED)
		  MESSAGE(FATAL_ERROR "Generator not compatible: ${CMAKE_GENERATOR}")
		ENDIF(MATLAB_FIND_REQUIRED)
	  ENDIF(${CMAKE_GENERATOR} MATCHES "Visual Studio")
	  
	  EXECUTE_PROCESS(COMMAND "${MATLAB_ROOT}/bin/mexext.bat" OUTPUT_VARIABLE MATLAB_MEXFILE_EXT OUTPUT_STRIP_TRAILING_WHITESPACE)
      
		FIND_LIBRARY(MATLAB_MEX_LIBRARY
		libmex
		${MATLAB_LIB_DIR}
		NO_DEFAULT_PATH
		)

		FIND_LIBRARY(MATLAB_MX_LIBRARY
		libmx
		${MATLAB_LIB_DIR}
		NO_DEFAULT_PATH
		)
        
		FIND_LIBRARY(MATLAB_ENG_LIBRARY
		libeng
		${MATLAB_LIB_DIR}
		NO_DEFAULT_PATH
		)

	ELSE( WIN32 )
	       SET(_MATLAB_ROOT_LST
		  $ENV{MATLABDIR}
		  $ENV{MATLAB_DIR}
		  /usr/local/matlab-7sp1/
		  /opt/matlab-7sp1/
		  $ENV{HOME}/matlab-7sp1/
		  $ENV{HOME}/redhat-matlab/
		  /usr/local/MATLAB/R2012b/
		  /usr/local/MATLAB/R2012a/
		  /usr/local/MATLAB/R2011b/
		  /usr/local/MATLAB/R2011a/
		  /usr/local/MATLAB/R2010bSP1/
		  /usr/local/MATLAB/R2010b/
		  /usr/local/MATLAB/R2010a/
		  /usr/local/MATLAB/R2009bSP1/
		  /usr/local/MATLAB/R2009b/
		  /usr/local/MATLAB/R2009a/
		  /usr/local/MATLAB/R2008b/
		  /usr/local/MATLAB/R2008a/
		  /Applications/MATLAB_R2012b.app/
		  /Applications/MATLAB_R2012a.app/
		  /Applications/MATLAB_R2011b.app/
		  /Applications/MATLAB_R2011a.app/
		  /Applications/MATLAB_R2010bSP1.app/
		  /Applications/MATLAB_R2010b.app/
		  /Applications/MATLAB_R2010a.app/
		  /Applications/MATLAB_R2009bSP1.app/
		  /Applications/MATLAB_R2009b.app/
		  /Applications/MATLAB_R2009a.app/
		  /Applications/MATLAB_R2008b.app/
		  /Applications/MATLAB_R2008a.app/
		  )
		#SET (LOOP_VAR "")
		SET(_MATLAB_ROOT "bin-NOTFOUND")
		FOREACH(LOOP_VAR ${_MATLAB_ROOT_LST})
			FIND_PATH(_MATLAB_ROOT "bin" ${LOOP_VAR} NO_DEFAULT_PATH)
			IF(_MATLAB_ROOT)			
				SET(MATLAB_ROOT ${_MATLAB_ROOT})
				BREAK()
			ENDIF()
		ENDFOREACH()


	  IF(64BIT)
	       SET(_MATLAB_LIB_DIR "bin-NOTFOUND")

		# AMD64 (Linux):
		FIND_PATH(_MATLAB_LIB_DIR "glnxa64" ${MATLAB_ROOT}/bin/ NO_DEFAULT_PATH)
		IF(_MATLAB_LIB_DIR)			
			SET(MATLAB_LIB_DIR ${_MATLAB_LIB_DIR}/glnxa64)
		ENDIF()
	
	   # Intel64 (OSX):
		FIND_PATH(_MATLAB_LIB_DIR "maci64" ${MATLAB_ROOT}/bin/ NO_DEFAULT_PATH)
		IF(_MATLAB_LIB_DIR)
			SET(MATLAB_LIB_DIR ${_MATLAB_LIB_DIR}/maci64)
		ENDIF()

		EXECUTE_PROCESS(COMMAND "${MATLAB_ROOT}/bin/mexext" OUTPUT_VARIABLE MATLAB_MEXFILE_EXT OUTPUT_STRIP_TRAILING_WHITESPACE)		
	  ELSE(64BIT)
		# Regular x86
		SET(MATLAB_LIB_DIR "${MATLAB_ROOT}/bin/glnx86/")
		EXECUTE_PROCESS(COMMAND "${MATLAB_ROOT}/bin/mexext" OUTPUT_VARIABLE MATLAB_MEXFILE_EXT OUTPUT_STRIP_TRAILING_WHITESPACE)
	  ENDIF(64BIT)
      FIND_LIBRARY(MATLAB_MEX_LIBRARY
		mex
		${MATLAB_LIB_DIR}
        	NO_DEFAULT_PATH
		)
      FIND_LIBRARY(MATLAB_MX_LIBRARY
		mx
		${MATLAB_LIB_DIR}
        	NO_DEFAULT_PATH
		)
      FIND_LIBRARY(MATLAB_ENG_LIBRARY
		eng
		${MATLAB_LIB_DIR}
        	NO_DEFAULT_PATH
		)
	  
	ENDIF(WIN32)
ENDIF(NOT MATLAB_ROOT)

SET(MATLAB_INCLUDE_DIR "${MATLAB_ROOT}/extern/include/")

# This is common to UNIX and Win32:
SET(MATLAB_LIBRARIES
  ${MATLAB_MEX_LIBRARY}
  ${MATLAB_MX_LIBRARY}
  ${MATLAB_ENG_LIBRARY}
)

IF(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)
  SET(MATLAB_FOUND 1)
ENDIF(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)

MARK_AS_ADVANCED(
  MATLAB_LIBRARIES
  MATLAB_MEX_LIBRARY
  MATLAB_MX_LIBRARY
  MATLAB_ENG_LIBRARY
  MATLAB_INCLUDE_DIR
  MATLAB_FOUND
  MATLAB_ROOT
  MATLAB_MEXFILE_EXT
)

