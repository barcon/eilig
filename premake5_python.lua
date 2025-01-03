-- premake5.lua
workspace "eilig_python"
	configurations { "Python"}
	location "build"

project "eilig_python"
	kind "SharedLib"
	language "C++"
	cppdialect "C++17"
	architecture "x86_64"	
	targetname ("_eilig")
	targetextension (".pyd")

	targetdir "build/%{cfg.buildcfg}"
	includedirs { "../utils/src" }
	includedirs { "../logger/src" }
	
	files "src/eilig.hpp"	
	files "src/eilig_matrix.hpp"	
	files "src/eilig_matrix.cpp"	
	files "src/eilig_matrix_ellpack.hpp"	
	files "src/eilig_matrix_ellpack.cpp"	
	files "src/eilig_routines.hpp"	
	files "src/eilig_routines.cpp"	
	files "src/eilig_status.hpp"	
	files "src/eilig_transform.hpp"	
	files "src/eilig_transform.cpp"	
	files "src/eilig_types.hpp"	
	files "src/eilig_vector.hpp"	
	files "src/eilig_vector.cpp"	

	filter "configurations:Python"
		kind "SharedLib"
		defines { "NDEBUG" }
		optimize "Speed"
		
		includedirs { "../python/inc" }
		
		files "src/eilig_export_python.cpp"	
		
		links { "utils", "logger", "python312" }
		
		libdirs { "../utils/build/Release" }
		libdirs { "../logger/build/Release" }
		libdirs { "../python/lib" }
		
workspace "eilig_pythonCL"
	configurations {"PythonCL"}
	location "build"

project "eilig_pythonCL"
	kind "SharedLib"
	language "C++"
	cppdialect "C++17"
	architecture "x86_64"	
	targetname ("_eiligCL")
	targetextension (".pyd")

	targetdir "build/%{cfg.buildcfg}"
	includedirs { "../utils/src" }
	includedirs { "../logger/src" }
	
	files "src/eilig.hpp"	
	files "src/eilig_matrix.hpp"	
	files "src/eilig_matrix.cpp"	
	files "src/eilig_matrix_ellpack.hpp"	
	files "src/eilig_matrix_ellpack.cpp"	
	files "src/eilig_routines.hpp"	
	files "src/eilig_routines.cpp"	
	files "src/eilig_status.hpp"	
	files "src/eilig_transform.hpp"	
	files "src/eilig_transform.cpp"	
	files "src/eilig_types.hpp"	
	files "src/eilig_vector.hpp"	
	files "src/eilig_vector.cpp"	
	
	filter "configurations:PythonCL"
		kind "SharedLib"
		defines { "NDEBUG", "EILIG_ENABLE_OPENCL" }
		optimize "Speed"
		
		includedirs { "../club/src" }		
		includedirs { "../opencl/inc" }
		includedirs { "../python/inc" }

		files "src/eilig_export_python_OpenCL.cpp"	
		
		files "src/eilig_opencl_entry_proxy.hpp"	
		files "src/eilig_opencl_entry_proxy.cpp"	
		files "src/eilig_opencl_kernels.hpp"	
		files "src/eilig_opencl_kernels.cpp"	
		files "src/eilig_opencl_matrix_ellpack.hpp"	
		files "src/eilig_opencl_matrix_ellpack.cpp"	
		files "src/eilig_opencl_vector.hpp"	
		files "src/eilig_opencl_vector.cpp"			
		
		links { "utils", "logger", "club", "opencl", "python312" }
		
		libdirs { "../utils/build/Release" }
		libdirs { "../logger/build/Release" }
		libdirs { "../club/build/Release" }
		libdirs { "../opencl/lib/x86_64" }
		libdirs { "../python/lib" }			