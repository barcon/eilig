-- premake5.lua
workspace "eilig"
	configurations { "Debug", "Release", "ReleaseCL"}
	location "build"

project "eilig"
	kind "StaticLib"
	language "C++"
	cppdialect "C++17"
	architecture "x86_64"	
	objdir "%{cfg.location}/obj/%{cfg.platform}_%{cfg.buildcfg}"

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
	
	filter "configurations:Debug"
		defines { "DEBUG" }
		symbols "On"
		
		links { "utils", "logger"}
		
		libdirs { "../utils/build/Debug" }
		libdirs { "../logger/build/Debug" }		

	filter "configurations:Release"
		defines { "NDEBUG" }
		optimize "Speed"
		
		links { "utils", "logger"}
		
		libdirs { "../utils/build/Release" }
		libdirs { "../logger/build/Release" }		
		
	filter "configurations:ReleaseCL"	
		defines { "NDEBUG", "EILIG_ENABLE_OPENCL" }
		optimize "Speed"		
		
		includedirs { "../club/src" }		
		includedirs { "../opencl/inc" }

		files "src/eilig_opencl_entry_proxy.hpp"	
		files "src/eilig_opencl_entry_proxy.cpp"	
		files "src/eilig_opencl_kernels.hpp"	
		files "src/eilig_opencl_kernels.cpp"	
		files "src/eilig_opencl_matrix_ellpack.hpp"	
		files "src/eilig_opencl_matrix_ellpack.cpp"	
		files "src/eilig_opencl_vector.hpp"	
		files "src/eilig_opencl_vector.cpp"
		
		links { "utils", "logger", "club", "opencl" }
		
		libdirs { "../utils/build/Release" }
		libdirs { "../logger/build/Release" }
		libdirs { "../club/build/Release" }
		libdirs { "../opencl/lib/x86_64" }